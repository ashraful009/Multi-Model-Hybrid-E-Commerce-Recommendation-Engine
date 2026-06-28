import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

def normalize_scores(scores):
    if len(scores) == 0: return []
    min_val, max_val = min(scores), max(scores)
    if max_val == min_val: return [0.5 for _ in scores]
    return [(x - min_val) / (max_val - min_val) for x in scores]

class BaselineModel:
    def __init__(self, knn_k=30, top_k=5):
        self.knn_k = knn_k
        self.top_k = top_k
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        self.knn = NearestNeighbors(n_neighbors=knn_k, metric='cosine', algorithm='brute')
        
    def train(self, all_products_df, train_df, all_products_list):
        print(" Building TF-IDF (Content-Based) Model...")
        self.tfidf_matrix = self.tfidf.fit_transform(all_products_df['combined_text'])
        
        
        print(" Building KNN (Collaborative Filtering) via Scikit-Learn...")
        # Create User-Item matrix for scikit-learn KNN
        self.user_item_matrix = train_df.pivot_table(index='userId', columns='productId', values='rating', fill_value=0)
        self.user_item_matrix = self.user_item_matrix.reindex(columns=all_products_list, fill_value=0)
        self.knn.fit(self.user_item_matrix.values)
        
        self.all_products_list = all_products_list
        self.product_idx_map = {pid: i for i, pid in enumerate(all_products_df['productId'])}
        self.all_products_df = all_products_df
        self.train_df = train_df

    def get_recommendations(self, target_user, top_k=None):
        if top_k is None:
            top_k = self.top_k
            
        # TF-IDF Score
        user_history = self.train_df[self.train_df['userId'] == target_user]['productId'].tolist()
        user_tfidf_profile = None
        if user_history:
            valid_indices = [self.product_idx_map[pid] for pid in user_history if pid in self.product_idx_map]
            if valid_indices:
                user_tfidf_profile = self.tfidf_matrix[valid_indices].mean(axis=0)

        tfidf_scores_raw = []
        for pid in self.all_products_list:
            if user_tfidf_profile is not None and pid in self.product_idx_map:
                p_idx = self.product_idx_map[pid]
                sim = cosine_similarity(np.asarray(user_tfidf_profile), self.tfidf_matrix[p_idx])[0][0]
                tfidf_scores_raw.append(sim)
            else:
                tfidf_scores_raw.append(0.0)

        # KNN Score
        knn_scores_raw = np.zeros(len(self.all_products_list))
        if target_user in self.user_item_matrix.index:
            user_idx = self.user_item_matrix.index.get_loc(target_user)
            user_vector = self.user_item_matrix.values[user_idx].reshape(1, -1)
            distances, indices = self.knn.kneighbors(user_vector, n_neighbors=min(self.knn_k, len(self.user_item_matrix)))
            neighbor_ratings = self.user_item_matrix.values[indices[0]]
            avg_ratings = neighbor_ratings.mean(axis=0)
            knn_scores_raw = avg_ratings

        tfidf_norm = normalize_scores(tfidf_scores_raw)
        knn_norm = normalize_scores(knn_scores_raw)

        results = []
        for i, pid in enumerate(self.all_products_list):
            final_score = (0.5 * tfidf_norm[i]) + (0.5 * knn_norm[i])
            results.append({'productId': pid, 'Final_Score': final_score})

        results_df = pd.DataFrame(results).sort_values(by='Final_Score', ascending=False)
        
        # Filter purchased items
        purchased = self.train_df[(self.train_df['userId'] == target_user) & (self.train_df['action_type'] == 'purchase')]['productId'].tolist()
        results_df = results_df[~results_df['productId'].isin(purchased)]
        
        return pd.merge(results_df.head(top_k), self.all_products_df[['productId', 'product_name', 'category']], on='productId', how='left')
