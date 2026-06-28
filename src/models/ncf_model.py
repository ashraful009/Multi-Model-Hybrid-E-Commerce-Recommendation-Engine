import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def generate_negative_samples(X_user, X_item, y, num_items, n_neg=4):
    print(f"Generating {n_neg} negative samples per positive interaction...")
    positive_pairs = set(zip(X_user.tolist(), X_item.tolist()))
    neg_users, neg_items, neg_labels = [], [], []
    
    for u, i in zip(X_user, X_item):
        count = 0
        while count < n_neg:
            neg_item = np.random.randint(0, num_items)
            if (int(u), neg_item) not in positive_pairs:
                neg_users.append(int(u))
                neg_items.append(neg_item)
                neg_labels.append(0.0)
                count += 1
    
    all_users = np.concatenate([X_user, np.array(neg_users)])
    all_items = np.concatenate([X_item, np.array(neg_items)])
    all_labels = np.concatenate([y, np.array(neg_labels)])
    
    indices = np.random.permutation(len(all_users))
    return all_users[indices], all_items[indices], all_labels[indices]

class NCFModel:
    def __init__(self, num_users, num_items, embedding_dim=64, top_k=5):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self.model = self._build_model()
        
    def _build_model(self):
        user_input = Input(shape=(1,), name="User_Input")
        user_embedding = Embedding(input_dim=self.num_users, output_dim=self.embedding_dim)(user_input)
        user_flat = Flatten()(user_embedding)

        item_input = Input(shape=(1,), name="Item_Input")
        item_embedding = Embedding(input_dim=self.num_items, output_dim=self.embedding_dim)(item_input)
        item_flat = Flatten()(item_embedding)

        concat = Concatenate()([user_flat, item_flat])

        dense_1 = Dense(128, activation='relu')(concat)
        bn_1 = BatchNormalization()(dense_1)
        drop_1 = Dropout(0.3)(bn_1)

        dense_2 = Dense(64, activation='relu')(drop_1)
        bn_2 = BatchNormalization()(dense_2)
        drop_2 = Dropout(0.2)(bn_2)

        dense_3 = Dense(32, activation='relu')(drop_2)
        dense_4 = Dense(16, activation='relu')(dense_3)

        output = Dense(1, activation='linear')(dense_4)

        ncf_model = Model(inputs=[user_input, item_input], outputs=output)
        ncf_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return ncf_model

    def train(self, train_df, batch_size=128, epochs=20, n_neg=4):
        tf.random.set_seed(42)
        
        X_user_train = train_df['user_encoded'].values
        X_item_train = train_df['item_encoded'].values
        y_train = train_df['rating'].values

        X_user_aug, X_item_aug, y_aug = generate_negative_samples(X_user_train, X_item_train, y_train, self.num_items, n_neg=n_neg)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
        ]

        print(" Training Enhanced NCF Model...")
        history = self.model.fit(
            [X_user_aug, X_item_aug], y_aug,
            batch_size=batch_size, epochs=epochs, verbose=1,
            validation_split=0.15,
            callbacks=callbacks
        )
        self.train_df = train_df
        return history

    def get_recommendations(self, target_user, all_products_list, all_products_df, user_encoder, product_idx_map, top_k=None):
        if top_k is None:
            top_k = self.top_k
            
        try:
            target_user_encoded = user_encoder.transform([target_user])[0]
        except ValueError:
            return pd.DataFrame()

        purchased = self.train_df[(self.train_df['userId'] == target_user) & (self.train_df['action_type'] == 'purchase')]['productId'].tolist()
        candidate_items = [pid for pid in all_products_list if pid not in purchased]

        if not candidate_items:
            return pd.DataFrame()

        candidate_encoded = []
        valid_candidates = []
        for pid in candidate_items:
            if pid in product_idx_map:
                candidate_encoded.append(product_idx_map[pid])
                valid_candidates.append(pid)

        user_input_array = np.array([target_user_encoded] * len(valid_candidates))
        item_input_array = np.array(candidate_encoded)

        preds = self.model.predict([user_input_array, item_input_array], batch_size=512, verbose=0).flatten()

        results_df = pd.DataFrame({'productId': valid_candidates, 'Predicted_Rating': preds})
        results_df = results_df.sort_values(by='Predicted_Rating', ascending=False).head(top_k)

        return pd.merge(results_df, all_products_df[['productId', 'product_name', 'category']], on='productId', how='left')
