import os
import sys
import numpy as np
import pandas as pd

def evaluate_model(model_name, recommend_func, test_df, top_k=5):
    print(f"\n📊 Evaluating {model_name}...")
    test_users = test_df['userId'].unique()
    
    total_precision = total_recall = total_ndcg = 0
    total_exact = total_cat = total_mrr = 0
    total_f1 = total_ap = total_hit = 0
    valid_users = 0
    
    # Data for confusion matrix and ROC
    all_pred_categories = []
    all_actual_categories = []
    all_relevance_scores = []
    all_relevance_labels = []
    
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    try:
        for user in test_users:
            actual_items_data = test_df[test_df['userId'] == user]
            actual_items = actual_items_data['productId'].tolist()
            actual_categories = actual_items_data['category'].dropna().unique().tolist()
            
            if not actual_items: continue
            valid_users += 1
            
            try:
                pred_recs = recommend_func(user)
                if pred_recs is None or pred_recs.empty:
                    continue
                pred_items = pred_recs['productId'].tolist()
                pred_categories = pred_recs['category'].dropna().tolist()
            except:
                continue
            
            # Collect category data for confusion matrix
            for cat in pred_categories:
                all_pred_categories.append(cat)
                if cat in actual_categories:
                    all_actual_categories.append(cat)
                else:
                    all_actual_categories.append(actual_categories[0] if actual_categories else 'Unknown')
            
            # Collect relevance data for ROC
            for item in pred_items:
                is_relevant = 1 if item in actual_items else 0
                all_relevance_labels.append(is_relevant)
                if item in pred_items:
                    rank = pred_items.index(item) + 1
                    all_relevance_scores.append(1.0 / rank)
            
            hits = len(set(pred_items) & set(actual_items))
            
            # Precision@K
            prec_k = hits / top_k
            total_precision += prec_k
            total_exact += prec_k
            
            # Recall@K
            rec_k = hits / len(actual_items)
            total_recall += rec_k
            
            # F1@K
            if prec_k + rec_k > 0:
                total_f1 += 2 * (prec_k * rec_k) / (prec_k + rec_k)
            
            # Hit Rate@K
            total_hit += 1 if hits > 0 else 0
            
            # Category Hit Rate
            cat_hits = sum(1 for cat in pred_categories if cat in actual_categories)
            total_cat += cat_hits / top_k
            
            # MRR
            for rank, item in enumerate(pred_items, 1):
                if item in actual_items:
                    total_mrr += 1.0 / rank
                    break
            
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
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

    n = valid_users if valid_users > 0 else 1
    results = {
        'Model': model_name,
        'Precision@K': round(total_precision / n, 4),
        'Recall@K': round(total_recall / n, 4),
        'F1@K': round(total_f1 / n, 4),
        'NDCG@K': round(total_ndcg / n, 4),
        'MAP@K': round(total_ap / n, 4),
        'Hit_Rate@K': round(total_hit / n, 4),
        'Exact_Match_Precision': round(total_exact / n, 4),
        'Category_Hit_Rate': round(total_cat / n, 4),
        'MRR': round(total_mrr / n, 4),
        '_pred_categories': all_pred_categories,
        '_actual_categories': all_actual_categories,
        '_relevance_scores': all_relevance_scores,
        '_relevance_labels': all_relevance_labels,
    }
    
    print(f"   ➤ Precision@{top_k}:  {results['Precision@K']:.4f}")
    print(f"   ➤ Recall@{top_k}:     {results['Recall@K']:.4f}")
    print(f"   ➤ F1@{top_k}:         {results['F1@K']:.4f}")
    print(f"   ➤ NDCG@{top_k}:       {results['NDCG@K']:.4f}")
    print(f"   ➤ MAP@{top_k}:        {results['MAP@K']:.4f}")
    print(f"   ➤ Hit Rate@{top_k}:   {results['Hit_Rate@K']:.4f}")
    print(f"   ➤ Category Hit Rate:  {results['Category_Hit_Rate']:.4f}")
    print(f"   ➤ MRR:               {results['MRR']:.4f}")
    return results
