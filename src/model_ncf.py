import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
from . import config

class NCFModel:
    def __init__(self, num_users, num_items, embedding_dim=config.NCF_EMBEDDING_DIM):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.model = None
        self.history = None

    def build_model(self):
        print("Building NCF Two-Tower model (Enhanced)...")
        tf.random.set_seed(42)
        
        # User Tower
        user_input = Input(shape=(1,), name="User_Input")
        user_embedding = Embedding(input_dim=self.num_users, output_dim=self.embedding_dim, name="User_Embedding")(user_input)
        user_flat = Flatten(name="User_Flatten")(user_embedding)

        # Item Tower
        item_input = Input(shape=(1,), name="Item_Input")
        item_embedding = Embedding(input_dim=self.num_items, output_dim=self.embedding_dim, name="Item_Embedding")(item_input)
        item_flat = Flatten(name="Item_Flatten")(item_embedding)

        # Fusion with BatchNorm and more Dropout
        concat = Concatenate(name="Tower_Fusion")([user_flat, item_flat])
        
        dense_1 = Dense(128, activation='relu')(concat)
        bn_1 = BatchNormalization()(dense_1)
        drop_1 = Dropout(0.3)(bn_1)
        
        dense_2 = Dense(64, activation='relu')(drop_1)
        bn_2 = BatchNormalization()(dense_2)
        drop_2 = Dropout(0.2)(bn_2)
        
        dense_3 = Dense(32, activation='relu')(drop_2)
        dense_4 = Dense(16, activation='relu')(dense_3)
        
        output = Dense(1, activation='linear', name="Prediction")(dense_4)

        self.model = Model(inputs=[user_input, item_input], outputs=output)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.NCF_LR),
            loss='mse',
            metrics=['mae']
        )

    @staticmethod
    def generate_negative_samples(X_user, X_item, y, num_items, n_neg=config.NCF_NEGATIVE_SAMPLES):
        """Generate negative samples for implicit feedback learning."""
        print(f"Generating {n_neg} negative samples per positive interaction...")
        positive_pairs = set(zip(X_user, X_item))
        
        neg_users = []
        neg_items = []
        neg_labels = []
        
        for u, i in zip(X_user, X_item):
            count = 0
            while count < n_neg:
                neg_item = np.random.randint(0, num_items)
                if (u, neg_item) not in positive_pairs:
                    neg_users.append(u)
                    neg_items.append(neg_item)
                    neg_labels.append(0.0)
                    count += 1
        
        all_users = np.concatenate([X_user, np.array(neg_users)])
        all_items = np.concatenate([X_item, np.array(neg_items)])
        all_labels = np.concatenate([y, np.array(neg_labels)])
        
        # Shuffle
        indices = np.random.permutation(len(all_users))
        return all_users[indices], all_items[indices], all_labels[indices]

    def train(self, X_user, X_item, y, epochs=config.NCF_EPOCHS, batch_size=config.NCF_BATCH_SIZE, use_negative_sampling=True):
        if use_negative_sampling:
            X_user, X_item, y = self.generate_negative_samples(
                X_user, X_item, y, self.num_items
            )
        
        print(f"Training NCF model for {epochs} epochs on {len(y)} samples...")
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
        ]
        
        self.history = self.model.fit(
            [X_user, X_item], y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_split=0.15,
            callbacks=callbacks
        )
        return self.history

    def save(self, path=config.NCF_MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"NCF model saved to {path}")

    def load(self, path=config.NCF_MODEL_PATH):
        self.model = load_model(path)
        print(f"NCF model loaded from {path}")

    def recommend(self, target_user, target_user_encoded, item_encoder, all_products_df, train_df, top_k=config.TOP_K, filter_purchased=True):
        all_encoded_items = np.array(range(self.num_items))
        user_input_array = np.full(self.num_items, target_user_encoded)

        preds = self.model.predict([user_input_array, all_encoded_items], verbose=0).flatten()
        
        # Sort all predictions
        sorted_indices = preds.argsort()[::-1]
        pred_items = item_encoder.inverse_transform(sorted_indices)
        
        results = []
        for pid in pred_items:
            prod_info = all_products_df[all_products_df['productId'] == pid].iloc[0]
            results.append({
                'productId': pid,
                'product_name': prod_info['product_name'],
                'category': prod_info['category']
            })
            
        results_df = pd.DataFrame(results)
        
        if filter_purchased:
            purchased_items = train_df[(train_df['userId'] == target_user) & 
                                     (train_df['action_type'] == 'purchase')]['productId'].tolist()
            out_of_stock_items = train_df[train_df['in_stock'] == 0]['productId'].tolist()
            
            results_df = results_df[~results_df['productId'].isin(purchased_items)]
            results_df = results_df[~results_df['productId'].isin(out_of_stock_items)]

        return results_df.head(top_k)
