import os
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from .global_var import SHOP_DUPLICATE_SET, CORRECT_CITY_NAME
import gc
from itertools import product


class ETL:
    """
     A class to perform Extract, Transform, Load (ETL) operations on data.

    Attributes:
        path_to_data (str): Path to the directory containing the data.

    Method:
        process(): Execute the full ETL pipeline

    Example usage:
        path_to_data = r'./data'  # Specify the path to your data
        etl = ETL(path_to_data)
        data = etl.process()
    """
    def __init__(self, path_to_data):
        self.path_to_data = path_to_data
        self.shop_duplicate_set = SHOP_DUPLICATE_SET
        self.city_name_mapping = CORRECT_CITY_NAME
        self.data = {}

    def extract_data(self):
        """Load all data from files and store it in the class."""
        self.data['sales_train_data'] = pd.read_csv(os.path.join(self.path_to_data, "sales_train.csv"))
        self.data['test_data'] = pd.read_csv(os.path.join(self.path_to_data, "test.csv")).set_index('ID')
        self.data['items_data'] = pd.read_csv(os.path.join(self.path_to_data, "items.csv"))
        self.data['item_categories_data'] = pd.read_csv(os.path.join(self.path_to_data, "item_categories.csv"))
        self.data['shops_data'] = pd.read_csv(os.path.join(self.path_to_data, "shops.csv"))
        self.data['sub_data'] = pd.read_csv(os.path.join(self.path_to_data, "sample_submission.csv"))

    def remove_outliers(self):
        """Remove outliers from sales_train_data."""
        self.data['sales_train_data'] = self.data['sales_train_data'][
            (self.data['sales_train_data']['item_cnt_day'] < 1000) &
            (self.data['sales_train_data']['item_price'] < 300000) &
            (self.data['sales_train_data']['item_price'] > 0)
        ]

    def replace_duplicate_shop_ids(self):
        """Replace duplicate shop IDs in all relevant datasets."""
        for key in ['sales_train_data', 'test_data', 'shops_data']:
            for old_id, new_id in self.shop_duplicate_set.items():
                self.data[key].loc[self.data[key]['shop_id'] == old_id, 'shop_id'] = new_id

    def extract_shop_subcategories(self):
        """Extract subcategories for shops_data."""
        shops_data = self.data['shops_data']
        shops_data['city'] = shops_data['shop_name'].apply(lambda x: x.split(' ')[0])
        shops_data['shop_type'] = shops_data['shop_name'].apply(lambda x: x.split(' ')[1] if len(x.split(' ')) > 1 else 'unknown')

        shop_type_count = shops_data['shop_type'].value_counts()
        valid_shop_types = shop_type_count[shop_type_count >= 5].index.to_list()

        shops_data['shop_type'] = shops_data['shop_type'].apply(
            lambda x: x if x in valid_shop_types else 'other'
        )
        self.data['shops_data'] = shops_data

    def extract_category_subcategories(self):
        """Extract subcategories for item_categories_data."""
        categories_data = self.data['item_categories_data']
        categories_data['item_category_type'] = categories_data['item_category_name'].apply(lambda x: x.split(' ')[0])
        categories_data['item_category_type'] = categories_data['item_category_type'].replace(
            {'Игровые': 'Games', 'Аксессуары': 'Games'}
        )

        category_type_count = categories_data['item_category_type'].value_counts()
        valid_category_types = category_type_count[category_type_count > 5].index.to_list()

        categories_data['item_category_type'] = categories_data['item_category_type'].apply(
            lambda x: x if x in valid_category_types else 'other'
        )
        categories_data['category_sub_type'] = categories_data['item_category_name'].apply(
            lambda x: x.split('-')[1].strip() if '-' in x else x.split(' ')[-1]
        )
        self.data['item_categories_data'] = categories_data

    def calculate_revenue(self):
        """Create revenue feature for sales_train_data."""
        self.data['sales_train_data']['revenue'] = (
            self.data['sales_train_data']['item_price'] * self.data['sales_train_data']['item_cnt_day']
        )

    def correct_city_names(self):
        """Correct city names for shops_data."""
        self.data['shops_data']['city'] = self.data['shops_data']['city'].replace(self.city_name_mapping)

    def transform_test_data(self):
        """Transform test data into the appropriate format."""
        test_data = self.data['test_data']
        test_data['date_block_num'] = 34
        test_data['shop_id'] = test_data['shop_id'].astype('int8')
        test_data['item_id'] = test_data['item_id'].astype('int16')
        test_data['date_block_num'] = test_data['date_block_num'].astype('int8')
        test_data.reset_index(drop=True, inplace=True)
        self.data['test_data'] = test_data

    def encode_data(self, key, columns):
        """Encode categorical features using OrdinalEncoder."""
        encoder = OrdinalEncoder()
        self.data[key][columns] = encoder.fit_transform(self.data[key][columns])

    def transform(self):
        """Data transformation: cleaning, handling duplicates, updating identifiers."""
        self.remove_outliers()
        self.replace_duplicate_shop_ids()
        self.extract_shop_subcategories()
        self.correct_city_names()
        self.extract_category_subcategories()
        self.calculate_revenue()
        self.encode_data('shops_data', ['city', 'shop_type'])
        self.encode_data('item_categories_data', ['category_sub_type', 'item_category_type'])
        self.transform_test_data()

        # Remove duplicates and select specific columns  
        self.data['shops_data'] = self.data['shops_data'][['shop_id', 'city', 'shop_type']].drop_duplicates(subset=['shop_id'])
        self.data['item_categories_data'] = self.data['item_categories_data'][['item_category_id', 'item_category_type', 'category_sub_type']]
        self.data['items_data'] = self.data['items_data'][['item_id', 'item_category_id']]

        # Step 1: Matrix Construction Functions
    def generate_sales_matrix(self, df, cols):
        """
        Generates a matrix of combinations for 'date_block_num', 'shop_id', and 'item_id'.
        """
        matrix = []
        for i in range(34):
            sales = df[df['date_block_num'] == i]
            matrix.append(np.array(list(product([i], sales['shop_id'].unique(), sales['item_id'].unique())), dtype=np.int16))
        return np.vstack(matrix)


    def create_matrix_dataframe( matrix, cols):
        """
        Converts the generated matrix into a pandas DataFrame, and sorts the values.
        """
        matrix_df = pd.DataFrame(data=matrix, columns=cols)
    
        return matrix_df.sort_values(cols).reset_index(drop=True)


    def add_item_cnt_month(self, matrix_df, df, cols):
        """
        Merges item_cnt_month data into the matrix DataFrame and appends test data.
        """
        matrix_df = pd.merge(matrix_df, df.groupby(cols)['item_cnt_day'].sum().reset_index(), on=cols, how='left').fillna(0)
        matrix_df['item_cnt_month'] = matrix_df['item_cnt_day'].astype('float16')
        matrix_df.drop(columns=['item_cnt_day'], inplace=True)
        return matrix_df


    def append_test_data(self, matrix_df, test_data):
        """
        Appends test data to the matrix DataFrame.
        """
        matrix_df = pd.concat([matrix_df, test_data], ignore_index=True)
        matrix_df['item_cnt_month'] = matrix_df['item_cnt_month'].fillna(0)
        return matrix_df


    def merge_with_external_data(self, matrix_df, items_data, item_categories_data, shops_data):
        """
        Merges item, category, and shop data into the matrix DataFrame.
        """
        matrix_df = pd.merge(matrix_df, items_data, on='item_id', how='left')
        matrix_df = pd.merge(matrix_df, item_categories_data, on='item_category_id', how='left')
        matrix_df = pd.merge(matrix_df, shops_data, on='shop_id', how='left')
        return matrix_df
    
    
    
    def process(self):
        """Execute the full ETL pipeline."""
        self.extract_data()
        self.transform()

        # Step 1: Matrix Creation
        cols = ['date_block_num', 'shop_id', 'item_id']
        df1 = self.data['sales_train_data']  
        matrix = self.generate_sales_matrix(df1, cols)
        matrix_df = self.create_matrix_dataframe(matrix, cols)
        del matrix
        gc.collect()
        matrix_df = self.add_item_cnt_month(matrix_df, df1, cols)
        del df1
        gc.collect()
        df2 = self.data['test_data']
        matrix_df = self.append_test_data(matrix_df, df2)
        del df2
        gc.collect()
        df3= self.data['items_data']
        df4 = self.data['item_categories_data']
        df5 = self.data['shops_data']
        matrix_df = self.merge_with_external_data(matrix_df, df3, df4, df5 )
        del df3, df4, df5
        gc.collect()
        print('1 --------------------------------')
        
        return matrix_df, self.data['sales_train_data']





#path_to_data = r'./data'
#etl = ETL(path_to_data)
#processed_data= etl.process()



