import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from . import config

class DataPreprocessor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.all_products_df = None
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()

    def process(self):
        print(f"Loading data from {self.csv_path}...")
        self.df = pd.read_csv(self.csv_path)
        
        self.handle_nulls()
        self.assign_categories()
        self.assign_brands()
        self.create_combined_text()
        self.parse_timestamps()
        
        # We need all_products_df for later reference
        self.all_products_df = self.df[['productId', 'product_name', 'category', 'brand', 'combined_text']].drop_duplicates(subset=['productId']).reset_index(drop=True)
        
        self.encode_labels()
        
        train_df, test_df = self.split_train_test()
        
        print("Data preprocessing complete.")
        return train_df, test_df, self.all_products_df

    def handle_nulls(self):
        self.df['product_name'] = self.df['product_name'].fillna('')
        self.df['brand'] = self.df['brand'].fillna('')
        
        if 'rating' not in self.df.columns:
            self.df['rating'] = 3.0
        else:
            self.df['rating'] = self.df['rating'].fillna(3.0)

    def assign_categories(self):
        def _assign(row):
            text = (str(row['product_name']) + " " + str(row.get('search_query', ''))).lower()
            if any(word in text for word in ['phone', 'iphone', 'samsung galaxy', 'smartphone']):
                return 'Smartphones'
            elif any(word in text for word in ['kindle', 'e-reader', 'eink']):
                return 'E-Readers'
            elif any(word in text for word in ['mouse', 'keyboard', 'monitor', 'mac mini', 'dock']):
                return 'Computer Accessories'
            elif any(word in text for word in ['headphone', 'earbud', 'audio', 'noise canceling']):
                return 'Audio & Headphones'
            elif any(word in text for word in ['tv', 'remote', 'ethernet', 'stick']):
                return 'Electronics Accessories'
            elif any(word in text for word in ['case', 'cover', 'skin', 'screen protector']):
                return 'Tech Accessories'
            elif any(word in text for word in ['battery', 'ssd', 'ram', 'cpu']):
                return 'Computer Parts'
            elif any(word in text for word in ['playstation', 'ps5', 'xbox', 'nintendo', 'game']):
                return 'Gaming'
            else:
                return 'General Electronics'
        
        # If category is largely missing, infer it
        if self.df['category'].isnull().mean() > 0.5:
            self.df['category'] = self.df.apply(_assign, axis=1)
        else:
            self.df['category'] = self.df['category'].fillna('')

    def assign_brands(self):
        def _assign(row):
            current_brand = str(row['brand'])
            if current_brand and current_brand != 'nan' and current_brand != 'Mock_brand':
                return current_brand

            text = str(row['product_name']).lower()
            known_brands = ['apple', 'samsung', 'sony', 'logitech', 'msi', 'dell', 'amazon', 'philips', 'meebook', 'uucovers']
            for b in known_brands:
                if b in text:
                    return b.capitalize()
            return 'Generic'
            
        self.df['brand'] = self.df.apply(_assign, axis=1)

    def create_combined_text(self):
        # Using standardized combined_text for all models
        self.df['combined_text'] = self.df['product_name'] + " " + self.df['category'] + " " + self.df['brand']

    def parse_timestamps(self):
        # Temporal Standardization: parse into naive Pandas datetime objects
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        # If timezone aware, convert to naive UTC
        if self.df['timestamp'].dt.tz is not None:
            self.df['timestamp'] = self.df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)

    def encode_labels(self):
        all_users = self.df['userId'].unique()
        all_items = self.df['productId'].unique()
        
        self.user_encoder.fit(all_users)
        self.item_encoder.fit(all_items)
        
        self.df['user_encoded'] = self.user_encoder.transform(self.df['userId'])
        self.df['item_encoded'] = self.item_encoder.transform(self.df['productId'])
        
        os.makedirs(os.path.dirname(config.USER_ENCODER_PATH), exist_ok=True)
        with open(config.USER_ENCODER_PATH, 'wb') as f:
            pickle.dump(self.user_encoder, f)
            
        with open(config.ITEM_ENCODER_PATH, 'wb') as f:
            pickle.dump(self.item_encoder, f)

    def split_train_test(self):
        # Sort by timestamp
        self.df = self.df.sort_values(by='timestamp')
        split_date = pd.to_datetime(config.SPLIT_DATE)
        
        train_df = self.df[self.df['timestamp'] < split_date]
        test_df = self.df[self.df['timestamp'] >= split_date]
        
        train_df.to_csv(config.TRAIN_DATA_PATH, index=False)
        test_df.to_csv(config.TEST_DATA_PATH, index=False)
        
        print(f"Train Set: {len(train_df)} rows")
        print(f"Test Set: {len(test_df)} rows")
        
        return train_df, test_df
