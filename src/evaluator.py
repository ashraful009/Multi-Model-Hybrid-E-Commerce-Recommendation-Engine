import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from . import config

class Evaluator:
    def __init__(self, test_df, train_df, all_products_df):
        self.test_df = test_df
        self.train_df = train_df
        self.all_products_df = all_products_df
        self.all_users = test_df['userId'].unique()
        self.product_idx_map = {pid: i for i, pid in enumerate(all_products_df['productId'])}
        
    def evaluate_model(self, model_name, recommend_func, product_embeddings=None, top_k=config.TOP_K):
        print(f"\nEvaluating {model_name} on {len(self.all_users)} test users...")
        
        total_precision = 0
        total_recall = 0
        total_ndcg = 0
        total_exact = 0
        total_cat = 0
        total_mrr = 0
        total_diversity = 0
        total_f1 = 0
        total_ap = 0
        total_hit = 0
        
        valid_users = 0
        recommended_unique_items = set()
        
        # For confusion matrix and ROC data collection
        all_pred_categories = []
        all_actual_categories = []
        all_relevance_scores = []
        all_relevance_labels = []
        
        # We suppress inner prints from recommend functions
        import sys
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        try:
            for user in self.all_users:
                actual_items_data = self.test_df[self.test_df['userId'] == user]
                actual_items = actual_items_data['productId'].tolist()
                actual_categories = actual_items_data['category'].dropna().unique().tolist()
                
                if not actual_items:
                    continue
                    
                valid_users += 1
                
                try:
                    pred_recs = recommend_func(user)
                    pred_items = pred_recs['productId'].tolist()
                    pred_categories = pred_recs['category'].dropna().tolist()
                except Exception as e:
                    pred_items = []
                    pred_categories = []
                
                recommended_unique_items.update(pred_items)
                
                # Collect category data for confusion matrix
                for cat in pred_categories:
                    all_pred_categories.append(cat)
                    if cat in actual_categories:
                        all_actual_categories.append(cat)
                    else:
                        # assign the most common actual category as the "expected"
                        all_actual_categories.append(actual_categories[0] if actual_categories else 'Unknown')
                
                # Collect relevance data for ROC curve
                for item in pred_items:
                    is_relevant = 1 if item in actual_items else 0
                    all_relevance_labels.append(is_relevant)
                    # Use position-based score (higher for top positions)
                    rank = pred_items.index(item) + 1
                    score = 1.0 / rank
                    all_relevance_scores.append(score)
                
                # Hits
                hits = len(set(pred_items) & set(actual_items))
                
                # Precision @ K
                precision_k = hits / top_k if top_k > 0 else 0
                total_precision += precision_k
                
                # Recall @ K
                recall_k = hits / len(actual_items) if actual_items else 0
                total_recall += recall_k
                
                # F1 @ K
                if precision_k + recall_k > 0:
                    f1_k = 2 * (precision_k * recall_k) / (precision_k + recall_k)
                else:
                    f1_k = 0
                total_f1 += f1_k
                
                # Hit Rate @ K (did at least 1 hit occur?)
                total_hit += 1 if hits > 0 else 0
                
                # Exact Match (same as Precision @ K but tracked specifically for Thesis)
                total_exact += hits / top_k if top_k > 0 else 0
                
                # Category Hit Rate
                cat_hits = sum(1 for cat in pred_categories if pd.notna(cat) and cat in actual_categories)
                total_cat += cat_hits / top_k if top_k > 0 else 0
                
                # MRR
                user_mrr = 0
                for rank, item in enumerate(pred_items, 1):
                    if item in actual_items:
                        user_mrr = 1.0 / rank
                        break
                total_mrr += user_mrr
                
                # Average Precision (for MAP)
                ap = 0
                num_hits = 0
                for rank, item in enumerate(pred_items, 1):
                    if item in actual_items:
                        num_hits += 1
                        ap += num_hits / rank
                if min(len(actual_items), top_k) > 0:
                    ap /= min(len(actual_items), top_k)
                total_ap += ap
                
                # NDCG
                dcg = 0
                idcg = sum([1.0 / np.log2(i + 1) for i in range(1, min(len(actual_items), top_k) + 1)])
                for rank, item in enumerate(pred_items, 1):
                    if item in actual_items:
                        dcg += 1.0 / np.log2(rank + 1)
                total_ndcg += dcg / idcg if idcg > 0 else 0
                
                # Diversity (if product_embeddings provided)
                if product_embeddings is not None and len(pred_items) > 1:
                    user_diversity = 0
                    pairs = 0
                    for i in range(len(pred_items)):
                        for j in range(i + 1, len(pred_items)):
                            pid1, pid2 = pred_items[i], pred_items[j]
                            if pid1 in self.product_idx_map and pid2 in self.product_idx_map:
                                idx1, idx2 = self.product_idx_map[pid1], self.product_idx_map[pid2]
                                vec1 = product_embeddings[idx1].reshape(1, -1)
                                vec2 = product_embeddings[idx2].reshape(1, -1)
                                sim = cosine_similarity(vec1, vec2)[0][0]
                                user_diversity += (1 - sim)
                                pairs += 1
                    total_diversity += (user_diversity / pairs) if pairs > 0 else 0
                    
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout
            
        avg_precision = total_precision / valid_users if valid_users > 0 else 0
        avg_recall = total_recall / valid_users if valid_users > 0 else 0
        avg_ndcg = total_ndcg / valid_users if valid_users > 0 else 0
        avg_exact = total_exact / valid_users if valid_users > 0 else 0
        avg_cat = total_cat / valid_users if valid_users > 0 else 0
        avg_mrr = total_mrr / valid_users if valid_users > 0 else 0
        avg_diversity = total_diversity / valid_users if valid_users > 0 else 0
        avg_f1 = total_f1 / valid_users if valid_users > 0 else 0
        avg_map = total_ap / valid_users if valid_users > 0 else 0
        hit_rate = total_hit / valid_users if valid_users > 0 else 0
        
        total_catalog_size = len(self.all_products_df)
        coverage = len(recommended_unique_items) / total_catalog_size if total_catalog_size > 0 else 0
        
        results = {
            'Model': model_name,
            'Precision@K': round(avg_precision, 4),
            'Recall@K': round(avg_recall, 4),
            'F1@K': round(avg_f1, 4),
            'NDCG@K': round(avg_ndcg, 4),
            'MAP@K': round(avg_map, 4),
            'Hit_Rate@K': round(hit_rate, 4),
            'Exact_Match_Precision': round(avg_exact, 4),
            'Category_Hit_Rate': round(avg_cat, 4),
            'MRR': round(avg_mrr, 4),
            'Catalog_Coverage': round(coverage, 4),
            'Intra_List_Diversity': round(avg_diversity, 4) if product_embeddings is not None else None,
            '_pred_categories': all_pred_categories,
            '_actual_categories': all_actual_categories,
            '_relevance_scores': all_relevance_scores,
            '_relevance_labels': all_relevance_labels,
        }
        
        self._print_results(results)
        return results

    def _print_results(self, results):
        print(f"\n[{results['Model']}] RESULTS:")
        print(f"  Precision@{config.TOP_K}: {results['Precision@K']:.4f}")
        print(f"  Recall@{config.TOP_K}:    {results['Recall@K']:.4f}")
        print(f"  F1@{config.TOP_K}:        {results['F1@K']:.4f}")
        print(f"  NDCG@{config.TOP_K}:      {results['NDCG@K']:.4f}")
        print(f"  MAP@{config.TOP_K}:       {results['MAP@K']:.4f}")
        print(f"  Hit Rate@{config.TOP_K}:  {results['Hit_Rate@K']:.4f}")
        print(f"  Exact Match:    {results['Exact_Match_Precision']:.4f}")
        print(f"  Category Hit:   {results['Category_Hit_Rate']:.4f}")
        print(f"  MRR:            {results['MRR']:.4f}")
        print(f"  Coverage:       {results['Catalog_Coverage'] * 100:.2f}%")
        if results['Intra_List_Diversity'] is not None:
            print(f"  Diversity:      {results['Intra_List_Diversity']:.4f}")

    def save_results(self, results_dict, filename):
        # Save without internal data columns
        save_dict = {k: v for k, v in results_dict.items() if not k.startswith('_')}
        df = pd.DataFrame([save_dict])
        path = os.path.join(config.METRICS_DIR, filename)
        df.to_csv(path, index=False)
        print(f"Saved results to {path}")

    def generate_comparison_table(self, results_list):
        clean_results = []
        for r in results_list:
            clean_results.append({k: v for k, v in r.items() if not k.startswith('_')})
        df = pd.DataFrame(clean_results)
        path = os.path.join(config.METRICS_DIR, 'master_comparison.csv')
        df.to_csv(path, index=False)
        print(f"Saved master comparison to {path}")
        return df
