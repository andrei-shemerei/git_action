import os
import re

import pandas as pd
import yaml




class ETL:
    """
    A class to perform Extract, Transform, Load (ETL) operations on data.

    Attributes:
        path_to_data (str): Path to the directory containing the data.

    Methods:
        extract(): Extracts data from the source.
        transform(): Transforms the extracted data.
        load(): Loads the transformed data into train and test DataFrames.

    Example usage:
        path_to_data = './data'  # Specify the path to your data
        etl = ETL(path_to_data)
        etl.extract()
        etl.transform()
        train_df, test_df = etl.load()
        print(train_df)
        print(test_df)
    """

    def __init__(self, path_to_data):
        self.path_to_data = path_to_data

    def extract(self):
        """Loading data from files."""
        self.sales_train_data = pd.read_csv(os.path.join(self.path_to_data, "sales_train.csv"))
        self.test_data = pd.read_csv(os.path.join(self.path_to_data, "test.csv"))
        self.items_data = pd.read_csv(os.path.join(self.path_to_data, "items.csv"))
        self.item_categories_data = pd.read_csv(os.path.join(self.path_to_data, "item_categories.csv"))
        self.shops_data = pd.read_csv(os.path.join(self.path_to_data, "shops.csv"))

    def _merge_data(self):
        """Merging data by identifiers."""
        for i, table in enumerate([self.sales_train_data, self.test_data]):
            for merge_table, merge_key in [
                (self.shops_data, "shop_id"),
                (self.items_data, "item_id"),
                (self.item_categories_data, "item_category_id")
            ]: 

                table = pd.merge(
                    table,
                    merge_table,
                    on=merge_key,
                    how="left"
                )

                if i == 0:
                    self.sales_train_data = table
                else:
                    self.test_data = table

    def _recount_and_clean_columns_in_data(self):
        """Grouping and cleaning data by key columns."""
        key_columns = self.sales_train_data.columns.tolist()
        key_columns.remove("item_cnt_day")

        self.sales_train_data = (
            self.sales_train_data.groupby(
            key_columns
            )["item_cnt_day"]
            .sum()
            .reset_index()
        )

        self.test_data = self.test_data.drop('ID', axis=1)

        # Converting date to datetime format
        self.sales_train_data["date"] = pd.to_datetime(
            self.sales_train_data["date"], dayfirst=True
        )

    def _normalize_data(self):
        """Normalization of shop and item names."""
        def normalize_name(name):
            """Function for string normalization."""
            name = re.sub(r"[^а-яё0-9a-z]", "", name.lower())
            name = re.sub(r"\s+", " ", name)
            return name

        def normalize_column(data, column):
            """Apply normalization to a specific column."""
            data[column] = data[column].apply(normalize_name)
            return data

        # Normalize shop names
        self.shops_data = normalize_column(self.shops_data, 'shop_name')

        # Normalize item names
        self.items_data = normalize_column(self.items_data, 'item_name')
    
    def _load_config(self, file_path):
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def transform(self):
        """Data transformation: cleaning, handling duplicates, updating identifiers."""
        def clean_data_with_negative_prices(data):
            """Removing rows with negative prices."""
            data = data[data.item_price >= 0]
            return data

        # Handling duplicates for shop data
        def concatenate_duplicate_shops(shops_data):
            base_dir = os.path.dirname(__file__)  
            config_path = os.path.join(base_dir, 'config.yaml')

            config = self._load_config(config_path)
            shop_id_duplicate_mapping = config['shop_id_mapping']
    
            # Update shop_id based on the mapping
            for old_id, new_id in shop_id_duplicate_mapping.items():
                shops_data.loc[shops_data.shop_id == old_id, "shop_id"] = new_id
                
            return shops_data

        def create_shop_mapping(shops_data):
            """Creating a shop_id mapping for duplicate removal."""
            shop_mapping = shops_data.index.to_series().map(shops_data['shop_id']).to_dict()
            return shop_mapping

        def create_item_mapping(items_data):
            """Creating an item_id mapping to handle duplicates by item_name."""
            items_data = items_data.drop_duplicates(subset=["item_name"]).reset_index(drop=True)
            item_mapping = self.items_data.set_index("item_name")["item_id"].to_dict()
            return item_mapping

        # Normalize data
        self._normalize_data()

        # Update shop data
        self.shops_data = concatenate_duplicate_shops(self.shops_data)

        # Update shop_id in test and training data
        shop_mapping = create_shop_mapping(self.shops_data)
        self.test_data['shop_id'] = self.test_data['shop_id'].map(shop_mapping)
        self.sales_train_data['shop_id'] = self.sales_train_data['shop_id'].map(shop_mapping)

        # Merge data
        self._merge_data()

        # Update item_id in test and training data
        item_mapping = create_item_mapping(self.items_data)
        self.sales_train_data["item_id"] = self.sales_train_data.item_name.map(item_mapping)
        self.test_data["item_id"] = self.test_data.item_name.map(item_mapping)

        # Group and clean data
        self._recount_and_clean_columns_in_data()

    def load(self):
        """Loading the transformed data."""
        train_df = self.sales_train_data
        test_df = self.test_data
        return train_df, test_df

