import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, KNNWithMeans
from . import config

class BaselineModel:
    def __init__(self, train_df, all_products_df):
        self.train_df = train_df
        self.all_products_df = all_products_df
        self.all_products_list = all_products_df['productId'].tolist()
        
        self.tfidf_matrix = None
        self.product_idx_map = None
        self.knn_model = None

    def build_tfidf(self):
        print("Building TF-IDF model...")
        tfidf = TfidfVectorizer(stop_words='english', max_features=config.TF_IDF_MAX_FEATURES)
        self.tfidf_matrix = tfidf.fit_transform(self.all_products_df['combined_text'])
        self.product_idx_map = {pid: i for i, pid in enumerate(self.all_products_df['productId'])}

    def build_knn(self):
        print("Building KNN collaborative filtering model (KNNWithMeans)...")
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.train_df[['userId', 'productId', 'rating']], reader)
        trainset = data.build_full_trainset()
        
        sim_options = {'name': 'pearson', 'user_based': True}
        self.knn_model = KNNWithMeans(k=config.KNN_K, sim_options=sim_options, verbose=False)
        self.knn_model.fit(trainset)

    def _normalize_scores(self, scores):
        if len(scores) == 0:
            return []
        min_val = min(scores)
        max_val = max(scores)
        if max_val == min_val:
            return [0.5 for _ in scores]
        return [(x - min_val) / (max_val - min_val) for x in scores]

    def recommend(self, target_user, top_k=config.TOP_K, filter_purchased=True):
        user_history = self.train_df[self.train_df['userId'] == target_user]['productId'].tolist()
        
        user_tfidf_profile = None
        if user_history:
            valid_indices = [self.product_idx_map[pid] for pid in user_history if pid in self.product_idx_map]
            if valid_indices:
                user_tfidf_profile = self.tfidf_matrix[valid_indices].mean(axis=0)

        tfidf_scores_raw = []
        knn_scores_raw = []
        
        for pid in self.all_products_list:
            if user_tfidf_profile is not None and pid in self.product_idx_map:
                p_idx = self.product_idx_map[pid]
                p_vector = self.tfidf_matrix[p_idx]
                sim = cosine_similarity(np.asarray(user_tfidf_profile), p_vector)[0][0]
                tfidf_scores_raw.append(sim)
            else:
                tfidf_scores_raw.append(0.0)

            cf_pred = self.knn_model.predict(target_user, pid).est
            knn_scores_raw.append(cf_pred)

        tfidf_norm = self._normalize_scores(tfidf_scores_raw)
        knn_norm = self._normalize_scores(knn_scores_raw)

        results = []
        for i, pid in enumerate(self.all_products_list):
            final_score = (0.5 * tfidf_norm[i]) + (0.5 * knn_norm[i])
            results.append({
                'productId': pid,
                'Final_Score': final_score,
                'TFIDF_Score': tfidf_norm[i],
                'KNN_Score': knn_norm[i]
            })

        results_df = pd.DataFrame(results).sort_values(by='Final_Score', ascending=False)
        
        if filter_purchased:
            purchased_items = self.train_df[(self.train_df['userId'] == target_user) & 
                                          (self.train_df['action_type'] == 'purchase')]['productId'].tolist()
            out_of_stock_items = self.train_df[self.train_df['in_stock'] == 0]['productId'].tolist()
            
            results_df = results_df[~results_df['productId'].isin(purchased_items)]
            results_df = results_df[~results_df['productId'].isin(out_of_stock_items)]

        final_output = pd.merge(results_df.head(top_k), 
                                self.all_products_df[['productId', 'product_name', 'category']], 
                                on='productId', how='left')
        return final_output
