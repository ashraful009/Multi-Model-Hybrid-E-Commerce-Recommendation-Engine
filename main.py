import pandas as pd
from src.data.data_processor import load_and_preprocess_data
from src.models.baseline_model import BaselineModel
from src.models.ncf_model import NCFModel
from src.models.hybrid_model import HybridModel
from src.evaluation.evaluator import evaluate_model
import warnings
import sys
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

def main():
    
    data = load_and_preprocess_data('dataset/ecommarce_full_dataset.csv')
    train_df = data['train_df']
    test_df = data['test_df']
    all_products_df = data['all_products_df']
    all_products_list = data['all_products_list']
    user_encoder = data['user_encoder']
    product_idx_map = data['product_idx_map']
    num_users = data['num_users']
    num_items = data['num_items']


    # 2. Baseline Model (TF-IDF + KNN)
    baseline = BaselineModel(knn_k=30, top_k=5)
    baseline.train(all_products_df, train_df, all_products_list)
    
    evaluate_model(
        model_name="Baseline (TF-IDF + KNN)",
        recommend_func=lambda u: baseline.get_recommendations(u, top_k=5),
        test_df=test_df,
        top_k=5
    )
    
    # 3. NCF Model (Deep Learning)
    ncf = NCFModel(num_users=num_users, num_items=num_items, embedding_dim=64, top_k=5)
    ncf.train(train_df, batch_size=128, epochs=20, n_neg=4) 
    
    evaluate_model(
        model_name="Deep Learning (NCF Two-Tower)",
        recommend_func=lambda u: ncf.get_recommendations(u, all_products_list, all_products_df, user_encoder, product_idx_map, top_k=5),
        test_df=test_df,
        top_k=5
    )
    
    # 4. Proposed Hybrid Model (MiniLM + SVD)
    hybrid = HybridModel(svd_n_factors=100, top_k=5, recent_items=5)
    hybrid.train(all_products_df, train_df, all_products_list)
    
    evaluate_model(
        model_name="Proposed Hybrid (MiniLM + SVD)",
        recommend_func=lambda u: hybrid.get_recommendations(u, top_k=5),
        test_df=test_df,
        top_k=5
    )
    
    print("\nPipeline Execution Complete.")

if __name__ == "__main__":
    main()
