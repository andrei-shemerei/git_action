import numpy as np
import pandas as pd
from itertools import product
from .global_var import FEATURE_SET
import gc

# Step 1: Matrix Construction Functions
def generate_sales_matrix(df, cols):
    """
    Generates a matrix of combinations for 'date_block_num', 'shop_id', and 'item_id'.
    """
    matrix = []
    for i in range(34):
        sales = df[df['date_block_num'] == i]
        matrix.append(np.array(list(product([i], sales['shop_id'].unique(), sales['item_id'].unique())), dtype=np.int16))
    return np.vstack(matrix)


def create_matrix_dataframe(matrix, cols):
    """
    Converts the generated matrix into a pandas DataFrame, and sorts the values.
    """
    matrix_df = pd.DataFrame(data=matrix, columns=cols)
    
    return matrix_df.sort_values(cols).reset_index(drop=True)


def add_item_cnt_month(matrix_df, df, cols):
    """
    Merges item_cnt_month data into the matrix DataFrame and appends test data.
    """
    matrix_df = pd.merge(matrix_df, df.groupby(cols)['item_cnt_day'].sum().reset_index(), on=cols, how='left').fillna(0)
    matrix_df['item_cnt_month'] = matrix_df['item_cnt_day'].astype('float16')
    matrix_df.drop(columns=['item_cnt_day'], inplace=True)
    return matrix_df


def append_test_data(matrix_df, test_data):
    """
    Appends test data to the matrix DataFrame.
    """
    matrix_df = pd.concat([matrix_df, test_data], ignore_index=True)
    matrix_df['item_cnt_month'] = matrix_df['item_cnt_month'].fillna(0)
    return matrix_df


def merge_with_external_data(matrix_df, items_data, item_categories_data, shops_data):
    """
    Merges item, category, and shop data into the matrix DataFrame.
    """
    matrix_df = pd.merge(matrix_df, items_data, on='item_id', how='left')
    matrix_df = pd.merge(matrix_df, item_categories_data, on='item_category_id', how='left')
    matrix_df = pd.merge(matrix_df, shops_data, on='shop_id', how='left')
    return matrix_df


# Step 2: Lag Feature Creation
def create_lag_feature(df, lags, features):
    for feature in features:
        for lag in lags:
            if f'{feature}_lag_{lag}' not in df.columns:
                df[f'{feature}_lag_{lag}'] = df.groupby(['shop_id', 'item_id'])[feature].shift(lag)
    return df


# Step 3: Creating Lags for Mean Sales Values
def merge_groupby_mean(df, group_cols, agg_col, new_col_name, dtype='float16'):
    if not isinstance(df, list):
        df = [df, df]

    df_process = 0
    df_with_info = 1
    
    temp = df[df_with_info].groupby(group_cols)[agg_col].mean().reset_index()
    temp.columns = group_cols + [new_col_name]
    df[df_process] = pd.merge(df[df_process], temp, on=group_cols, how='left')
    del temp
    gc.collect()
    df[df_process][new_col_name] = df[df_process][new_col_name].astype(dtype)
    return df[df_process]


def create_lags_feature_by_item_cnt_mean(df, feature_set, lags):
    df['item_cnt_month'] = df['item_cnt_month'].astype('float32')
    for feature, columns in feature_set.items():
        df = merge_groupby_mean(df, columns, 'item_cnt_month', feature)
        df = create_lag_feature(df, lags, [feature])
        df.drop(columns=feature, inplace=True)
    df['item_cnt_month'] = df['item_cnt_month'].astype('float16')
    return df


# Step 4: Processing Price Features
def process_price_features(df, sales_train_data, lags):
    df = merge_groupby_mean([df, sales_train_data], ['item_id'], 'item_price', 'item_id_price_mean')
    print('4.1-------------------')
    df = merge_groupby_mean([df, sales_train_data], ['date_block_num', 'item_id'], 'item_price', 'date_item_id_price_mean')
    print('4.2-------------------')
    del sales_train_data
    gc.collect()
    df = create_lag_feature(df, lags, ['date_item_id_price_mean'])
    print('4.3-------------------')

    for i in lags:
        df[f'delta_price_lag_{i}'] = (df[f'date_item_id_price_mean_lag_{i}'] - df['item_id_price_mean']) / df['item_id_price_mean']
        df[f'delta_price_lag_{i}'] = df[f'delta_price_lag_{i}'].astype('float16')
    print('4.4-------------------')
    df['nearest_delta_price_lag'] = df.apply(select_changes, axis=1, lags=lags)
    df['nearest_delta_price_lag'] = df['nearest_delta_price_lag'].astype('float16')
    print('4.5-------------------')
    drop_columns = ['item_id_price_mean', 'date_item_id_price_mean'] + [f'delta_price_lag_{i}' for i in lags]
    df.drop(columns=drop_columns, inplace=True)
    print('4.6-------------------')
    return df


def select_changes(row, lags):
    for i in lags:
        if row[f'delta_price_lag_{i}'] != 0:
            return row[f'delta_price_lag_{i}']
    return 0


# Step 5: Processing Revenue Features
def process_revenue_features(df, sales_train_data, lags):
    df = merge_groupby_mean([df, sales_train_data], ['shop_id'], 'revenue', 'shop_id_revenue_mean')
    df = merge_groupby_mean([df, sales_train_data], ['date_block_num', 'shop_id'], 'revenue', 'date_shop_id_revenue_mean')

    df['delta_revenue'] = (df['date_shop_id_revenue_mean'] - df['shop_id_revenue_mean']) / df['shop_id_revenue_mean']
    df['delta_revenue'] = df['delta_revenue'].astype('float16')

    df = create_lag_feature(df, lags, ['delta_revenue'])

    df.drop(columns=['delta_revenue', 'shop_id_revenue_mean', 'date_shop_id_revenue_mean'], inplace=True)

    return df


# Step 6: Adding Date Features and Filling NaN
def add_date_features(df):
    df['month'] = df['date_block_num'] % 12
    df['sezon'] = df['month'] // 4
    return df


def fill_na(df):
    for col in df.columns:
        if 'lag' in col and df[col].isnull().any():
            df[col] = df[col].fillna(0)
    return df


# Main pipeline
def main_pipeline(matrix_df, sales_train_data):
    """
    Call the pipeline with the required datasets
    For example: 
     final_df = main_pipeline(data)

    variable data shuld contain folow set of data:
        sales_train_data
        test_data
        items_data
        item_categories_data
        shops_data
    """    
    
    # Step 1: Matrix Creation
    cols = ['date_block_num', 'shop_id', 'item_id']

    #matrix = generate_sales_matrix(sales_train_data, cols)
    #matrix_df = create_matrix_dataframe(matrix, cols)
    #del matrix
    #gc.collect()
    #matrix_df = add_item_cnt_month(matrix_df, sales_train_data, cols)
    #matrix_df = append_test_data(matrix_df, test_data)
    #del test_data
    #gc.collect()
    #matrix_df = merge_with_external_data(matrix_df, items_data, item_categories_data, shops_data)
    #del items_data, item_categories_data, shops_data
    #gc.collect()
    #print('1 --------------------------------')
    # Step 2: Create lags for item_cnt_month
    lags = [1, 2, 3, 6, 12]
    train_test_df = create_lag_feature(matrix_df, lags, ['item_cnt_month'])
    del matrix_df
    gc.collect()
    print('2 ----------------------------------')    
    # Step 3: Create lags for mean sales values
    train_test_df = create_lags_feature_by_item_cnt_mean(train_test_df, FEATURE_SET, lags)
    print('3 ----------------------------------')  
    # Step 4: Process price features and create price lags
    train_test_df = process_price_features(train_test_df, sales_train_data, lags)
    print('4 ----------------------------------') 
    # Step 5: Process revenue features and create revenue lags
    train_test_df = process_revenue_features(train_test_df, sales_train_data, lags)
    del sales_train_data
    gc.collect()
    print('5 ----------------------------------')  
    # Step 6: Add date features and fill missing values
    train_test_df = add_date_features(train_test_df)
    train_test_df = fill_na(train_test_df)

    train_test_df = train_test_df[train_test_df.date_block_num > 11]
    print('6 ----------------------------------')
    return train_test_df



#from etl_process import ETL
#path_to_data = r'./data'
#etl = ETL(path_to_data)
#data = etl.process()
#for df in data:
#    print(data[df].shape)
#print(0)
#train_test_df = main_pipeline(data)
#print(train_test_df)
#
#print(train_test_df.shape)