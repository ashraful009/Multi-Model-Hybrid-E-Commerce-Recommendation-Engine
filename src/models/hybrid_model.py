import numpy as np
import pandas as pd
import math
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

def normalize_scores(scores):
    if len(scores) == 0: return []
    min_val, max_val = min(scores), max(scores)
    if max_val == min_val: return [0.5 for _ in scores]
    return [(x - min_val) / (max_val - min_val) for x in scores]

class HybridModel:
    def __init__(self, svd_n_factors=100, top_k=5, recent_items=5, decay_lambda=0.005, decay_alpha_min=0.3, decay_alpha_range=0.5):
        self.svd_n_factors = svd_n_factors
        self.top_k = top_k
        self.recent_items = recent_items
        self.decay_lambda = decay_lambda
        self.decay_alpha_min = decay_alpha_min
        self.decay_alpha_range = decay_alpha_range
        
    def train(self, all_products_df, train_df, all_products_list):
        print(" Building Semantic Engine (MiniLM)...")
        self.nlp_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.product_embeddings = self.nlp_model.encode(all_products_df['combined_text'].tolist(), show_progress_bar=True)
        
        print(" Building Collaborative Engine (Scikit-Learn TruncatedSVD)...")
        self.user_item_matrix_svd = train_df.pivot_table(index='userId', columns='productId', values='rating')
        global_mean = self.user_item_matrix_svd.stack().mean()
        self.user_item_matrix_svd = self.user_item_matrix_svd.fillna(global_mean)
        self.user_item_matrix_svd = self.user_item_matrix_svd.reindex(columns=all_products_list, fill_value=global_mean)
        
        n_components = min(self.svd_n_factors, len(self.user_item_matrix_svd) - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        user_factors = self.svd.fit_transform(self.user_item_matrix_svd.values)
        item_factors = self.svd.components_
        self.svd_reconstructed = np.dot(user_factors, item_factors)
        
        self.all_products_list = all_products_list
        self.product_idx_map = {pid: i for i, pid in enumerate(all_products_df['productId'])}
        self.all_products_df = all_products_df
        self.train_df = train_df
        
    def get_recommendations(self, target_user, top_k=None):
        if top_k is None:
            top_k = self.top_k
            
        user_history = self.train_df[self.train_df['userId'] == target_user]
        
        t_days = 180.0
        if not user_history.empty:
            last_seen = user_history['timestamp'].max()
            global_max = self.train_df['timestamp'].max()
            t_days = (global_max - last_seen).total_seconds() / 86400.0
            
        # Continuous Dynamic Weights
        alpha = self.decay_alpha_min + self.decay_alpha_range * math.exp(-self.decay_lambda * t_days)
        beta = 1.0 - alpha

        # Semantic Scores using top-N recent items
        semantic_scores = np.zeros(len(self.all_products_list))
        recent_items = user_history.sort_values('timestamp', ascending=False).head(self.recent_items)['productId'].tolist()
        
        if recent_items:
            recent_indices = [self.product_idx_map[pid] for pid in recent_items if pid in self.product_idx_map]
            if recent_indices:
                user_intent_vector = np.mean(self.product_embeddings[recent_indices], axis=0).reshape(1, -1)
                semantic_scores = cosine_similarity(user_intent_vector, self.product_embeddings)[0]

        semantic_norm = normalize_scores(semantic_scores)
        
        # SVD Collaborative Score
        cf_scores = np.zeros(len(self.all_products_list))
        if target_user in self.user_item_matrix_svd.index:
            user_idx = self.user_item_matrix_svd.index.get_loc(target_user)
            cf_scores = self.svd_reconstructed[user_idx]
            
        cf_norm = normalize_scores(cf_scores)

        results = []
        for i, pid in enumerate(self.all_products_list):
            final_score = (alpha * semantic_norm[i]) + (beta * cf_norm[i])
            results.append({'productId': pid, 'Final_Score': final_score})

        results_df = pd.DataFrame(results).sort_values('Final_Score', ascending=False)
        
        purchased = self.train_df[(self.train_df['userId'] == target_user) & (self.train_df['action_type'] == 'purchase')]['productId'].tolist()
        results_df = results_df[~results_df['productId'].isin(purchased)]
        
        return pd.merge(results_df.head(top_k), self.all_products_df[['productId', 'product_name', 'category']], on='productId', how='left')
