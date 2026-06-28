import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(dataset_path):
    print(f"📂 Loading Dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    # 1. Handle Nulls
    df['product_name'] = df['product_name'].fillna('')
    df['brand'] = df['brand'].fillna('')
    if 'rating' not in df.columns:
        df['rating'] = 3.0
    else:
        df['rating'] = df['rating'].fillna(3.0)

    # 2. Assign Categories (Inference)
    def _assign_cat(row):
        text = (str(row['product_name']) + " " + str(row.get('search_query', ''))).lower()
        if any(word in text for word in ['phone', 'iphone', 'samsung galaxy', 'smartphone']): return 'Smartphones'
        elif any(word in text for word in ['kindle', 'e-reader', 'eink']): return 'E-Readers'
        elif any(word in text for word in ['mouse', 'keyboard', 'monitor', 'mac mini', 'dock']): return 'Computer Accessories'
        elif any(word in text for word in ['headphone', 'earbud', 'audio', 'noise canceling']): return 'Audio & Headphones'
        elif any(word in text for word in ['tv', 'remote', 'ethernet', 'stick']): return 'Electronics Accessories'
        elif any(word in text for word in ['case', 'cover', 'skin', 'screen protector']): return 'Tech Accessories'
        elif any(word in text for word in ['battery', 'ssd', 'ram', 'cpu']): return 'Computer Parts'
        elif any(word in text for word in ['playstation', 'ps5', 'xbox', 'nintendo', 'game']): return 'Gaming'
        return 'General Electronics'

    if df['category'].isnull().mean() > 0.5:
        df['category'] = df.apply(_assign_cat, axis=1)
    else:
        df['category'] = df['category'].fillna('')

    # 3. Assign Brands
    def _assign_brand(row):
        current_brand = str(row['brand'])
        if current_brand and current_brand != 'nan' and current_brand != 'Mock_brand':
            return current_brand
        text = str(row['product_name']).lower()
        known_brands = ['apple', 'samsung', 'sony', 'logitech', 'msi', 'dell', 'amazon', 'philips', 'meebook', 'uucovers']
        for b in known_brands:
            if b in text:
                return b.capitalize()
        return 'Generic'

    df['brand'] = df.apply(_assign_brand, axis=1)

    # 4. Standardize text
    df['combined_text'] = df['product_name'] + " " + df['category'] + " " + df['brand']

    # 5. Timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if df['timestamp'].dt.tz is not None:
        df['timestamp'] = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)

    # 6. Label Encoding for Deep Learning
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    df['user_encoded'] = user_encoder.fit_transform(df['userId'])
    df['item_encoded'] = item_encoder.fit_transform(df['productId'])

    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)

    # 7. Train/Test Split (Temporal)
    df = df.sort_values(by='timestamp')
    split_date = pd.to_datetime('2025-12-01')
    train_df = df[df['timestamp'] < split_date].copy()
    test_df = df[df['timestamp'] >= split_date].copy()

    all_products_df = df[['productId', 'product_name', 'category', 'brand', 'combined_text']].drop_duplicates(subset=['productId']).reset_index(drop=True)
    all_products_list = all_products_df['productId'].tolist()
    product_idx_map = {pid: i for i, pid in enumerate(all_products_df['productId'])}

    print(f" Successfully processed {len(df)} total records.")
    
    return {
        'train_df': train_df,
        'test_df': test_df,
        'all_products_df': all_products_df,
        'all_products_list': all_products_list,
        'product_idx_map': product_idx_map,
        'user_encoder': user_encoder,
        'item_encoder': item_encoder,
        'num_users': num_users,
        'num_items': num_items
    }
