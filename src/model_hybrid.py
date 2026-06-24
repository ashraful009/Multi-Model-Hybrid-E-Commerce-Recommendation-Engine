import pandas as pd
import numpy as np
import math
import os
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from . import config

class HybridModel:
    def __init__(self, train_df, all_products_df):
        self.train_df = train_df
        self.all_products_df = all_products_df
        self.all_products_list = all_products_df['productId'].tolist()
        
        self.nlp_model = None
        self.product_embeddings = None
        self.product_idx_map = None
        self.svd_model = None

    def build_semantic_engine(self):
        print("Building Semantic Engine (MiniLM embeddings)...")
        self.nlp_model = SentenceTransformer(config.MINILM_MODEL_NAME)
        
        # We use the standardized combined_text from preprocessor
        texts = self.all_products_df['combined_text'].tolist()
        
        embeddings = self.nlp_model.encode(texts, show_progress_bar=True)
        self.product_embeddings = embeddings
        self.product_idx_map = {pid: i for i, pid in enumerate(self.all_products_list)}
        
        os.makedirs(os.path.dirname(config.EMBEDDINGS_PATH), exist_ok=True)
        with open(config.EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump({'embeddings': self.product_embeddings, 'idx_map': self.product_idx_map}, f)
        print("Product embeddings saved.")

    def load_semantic_engine(self):
        with open(config.EMBEDDINGS_PATH, 'rb') as f:
            data = pickle.load(f)
            self.product_embeddings = data['embeddings']
            self.product_idx_map = data['idx_map']
        print("Product embeddings loaded.")

    def build_collaborative_engine(self):
        print("Building Collaborative Engine (SVD with regularization)...")
        # Generate composite score
        action_weights = {'purchase': 5.0, 'cart': 3.0, 'view': 1.0}
        self.train_df['action_weight'] = self.train_df['action_type'].map(action_weights).fillna(1.0)
        
        max_duration = self.train_df['view_duration'].max()
        self.train_df['duration_score'] = (self.train_df['view_duration'] / max_duration) * 5.0
        
        self.train_df['composite_score'] = (self.train_df['action_weight'] * 0.5) + \
                                           (self.train_df['rating'] * 0.4) + \
                                           (self.train_df['duration_score'] * 0.1)
        
        user_item_matrix = self.train_df.groupby(['userId', 'productId'])['composite_score'].max().reset_index()
        
        reader = Reader(rating_scale=(0.0, 5.5))
        data = Dataset.load_from_df(user_item_matrix[['userId', 'productId', 'composite_score']], reader)
        trainset = data.build_full_trainset()
        
        self.svd_model = SVD(n_factors=config.SVD_N_FACTORS, 
                             n_epochs=config.SVD_N_EPOCHS, 
                             lr_all=config.SVD_LR,
                             reg_all=config.SVD_REG,
                             random_state=42)
        self.svd_model.fit(trainset)
        
        os.makedirs(os.path.dirname(config.SVD_MODEL_PATH), exist_ok=True)
        with open(config.SVD_MODEL_PATH, 'wb') as f:
            pickle.dump(self.svd_model, f)
        print("SVD model saved.")

    def load_collaborative_engine(self):
        with open(config.SVD_MODEL_PATH, 'rb') as f:
            self.svd_model = pickle.load(f)
        print("SVD model loaded.")

    def compute_dynamic_weights(self, t_days):
        """
        Continuous exponential decay for intent weights.
        """
        alpha = config.DECAY_ALPHA_MIN + config.DECAY_ALPHA_RANGE * math.exp(-config.DECAY_LAMBDA * t_days)
        beta = 1.0 - alpha
        return alpha, beta

    def _normalize_scores(self, scores):
        if len(scores) == 0:
            return []
        min_val = min(scores)
        max_val = max(scores)
        if max_val == min_val:
            return [0.5 for _ in scores]
        return [(x - min_val) / (max_val - min_val) for x in scores]

    def recommend(self, target_user, top_k=config.TOP_K, filter_purchased=True):
        user_history = self.train_df[self.train_df['userId'] == target_user]
        
        t_days = 180.0 # Default to max (inactive/cold start)
        if not user_history.empty:
            last_seen = user_history['timestamp'].max()
            global_max = self.train_df['timestamp'].max()
            t_days = (global_max - last_seen).total_seconds() / 86400.0
            
        alpha, beta = self.compute_dynamic_weights(t_days)

        semantic_scores = np.zeros(len(self.all_products_list))
        user_recent_items = user_history.sort_values('timestamp', ascending=False).head(config.HYBRID_RECENT_ITEMS)['productId'].tolist()
        
        if user_recent_items:
            recent_indices = [self.product_idx_map[pid] for pid in user_recent_items if pid in self.product_idx_map]
            if recent_indices:
                user_intent_vector = np.mean(self.product_embeddings[recent_indices], axis=0).reshape(1, -1)
                semantic_scores = cosine_similarity(user_intent_vector, self.product_embeddings)[0]

        semantic_norm = self._normalize_scores(semantic_scores)
        
        cf_scores = [self.svd_model.predict(target_user, pid).est for pid in self.all_products_list]
        cf_norm = self._normalize_scores(cf_scores)

        results = []
        for i, pid in enumerate(self.all_products_list):
            final_score = (alpha * semantic_norm[i]) + (beta * cf_norm[i])
            results.append({
                'productId': pid,
                'Final_Score': final_score,
                'Semantic_Score': semantic_norm[i],
                'CF_Score': cf_norm[i]
            })

        results_df = pd.DataFrame(results).sort_values('Final_Score', ascending=False)
        
        if filter_purchased:
            purchased_items = self.train_df[(self.train_df['userId'] == target_user) & 
                                          (self.train_df['action_type'] == 'purchase')]['productId'].tolist()
            out_of_stock_items = self.train_df[self.train_df['in_stock'] == 0]['productId'].tolist()
            
            results_df = results_df[~results_df['productId'].isin(purchased_items)]
            results_df = results_df[~results_df['productId'].isin(out_of_stock_items)]

        final_output = pd.merge(results_df.head(top_k), 
                                self.all_products_df[['productId', 'product_name', 'category']], 
                                on='productId', how='left')
        return final_output, t_days, alpha, beta
