import pandas as pd
import numpy as np
import random
from datetime import timedelta
import os

def generate_synthetic_data(csv_path):
    print("Loading original dataset...")
    df = pd.read_csv(csv_path)
    
    # Fill missing values temporarily for safe sampling
    df['category'] = df['category'].fillna('')
    df['brand'] = df['brand'].fillna('')
    
    # We will sample product info directly from the dataframe
    # Let's get unique products and their attributes
    product_cols = ['productId', 'product_name', 'category', 'brand', 'description', 'current_price', 'in_stock']
    products_df = df[product_cols].drop_duplicates(subset=['productId']).dropna(subset=['productId'])
    
    # Sample lists
    search_queries = df['search_query'].dropna().unique().tolist()
    if not search_queries:
        search_queries = ['default query']
    
    action_types = ['view', 'cart', 'purchase']
    action_probs = [0.7, 0.2, 0.1]
    
    # Time settings
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    global_max_time = df['timestamp'].max()
    
    print(f"Global max timestamp: {global_max_time}")
    
    new_rows = []
    
    existing_i_ids = set(df['interaction_id'].astype(str))
    existing_s_ids = set(df['session_id'].astype(str))
    
    def get_unique_id(prefix, existing_set):
        while True:
            new_id = f"{prefix}{random.randint(10000, 99999)}"
            if new_id not in existing_set:
                existing_set.add(new_id)
                return new_id

    print("Generating 30 synthetic users (U021 to U050)...")
    for i in range(21, 51):
        user_id = f"U{i:03d}"
        
        # Balance: Even IDs = Long-Term, Odd IDs = Short-Term
        is_short_term = (i % 2 != 0)
        
        num_interactions = random.randint(15, 40)
        
        for _ in range(num_interactions):
            # Pick a product
            product = products_df.sample(n=1).iloc[0]
            
            # Determine timestamp based on user profile
            if is_short_term:
                # Active in the last 7 days
                days_ago = random.uniform(0, 7)
            else:
                # Active between 30 and 180 days ago (Long-term)
                days_ago = random.uniform(30, 180)
            
            interact_time = global_max_time - timedelta(days=days_ago)
            
            # Action and Rating
            action = np.random.choice(action_types, p=action_probs)
            rating = round(random.uniform(1.0, 5.0), 1)
            if action == 'purchase':
                rating = round(random.uniform(3.5, 5.0), 1) # Purchases usually have higher ratings
                
            # Random discount
            discount = random.choice([0, 5, 10, 15, 20])
            
            new_rows.append({
                'interaction_id': get_unique_id('I', existing_i_ids),
                'userId': user_id,
                'productId': product['productId'],
                'session_id': get_unique_id('S', existing_s_ids),
                'action_type': action,
                'timestamp': interact_time.strftime('%Y-%m-%d %H:%M:%S'),
                'view_duration': random.randint(10, 300),
                'rating': rating,
                'product_name': product['product_name'],
                'category': product['category'],
                'brand': product['brand'],
                'description': product['description'],
                'search_query': random.choice(search_queries),
                'current_price': product['current_price'],
                'discount_percent': discount,
                'in_stock': product['in_stock']
            })

    new_df = pd.DataFrame(new_rows)
    print(f"Generated {len(new_df)} synthetic interactions.")
    
    # Restore actual NaNs where empty strings were placed for category and brand in original DF before saving
    # (Since the preprocessor handles inference)
    new_df['category'] = new_df['category'].replace('', np.nan)
    new_df['brand'] = new_df['brand'].replace('', np.nan)
    
    # Merge and save
    enhanced_df = pd.concat([df, new_df], ignore_index=True)
    
    # Restore original df NaNs before saving
    enhanced_df['category'] = enhanced_df['category'].replace('', np.nan)
    enhanced_df['brand'] = enhanced_df['brand'].replace('', np.nan)
    
    enhanced_df.to_csv(csv_path, index=False)
    print(f"Enhanced dataset saved to {csv_path} with {len(enhanced_df)} total records.")
    
    print("\nDataset Summary:")
    print(f"Total Users: {enhanced_df['userId'].nunique()}")
    
    # Check if short-term vs long-term is balanced
    enhanced_df['timestamp'] = pd.to_datetime(enhanced_df['timestamp'])
    max_t = enhanced_df['timestamp'].max()
    active_mask = (max_t - enhanced_df['timestamp']).dt.days <= 7
    active_users = enhanced_df[active_mask]['userId'].nunique()
    inactive_users = enhanced_df[~active_mask]['userId'].nunique()
    
    print(f"Users active in last 7 days: {active_users}")
    print(f"Users with older history: {inactive_users}")

if __name__ == '__main__':
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'ecommarce_full_dataset.csv')
    generate_synthetic_data(csv_path)
