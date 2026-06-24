import argparse
import sys
import os

# Ensure the working directory is set to the project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.data_preprocessor import DataPreprocessor
from src.visualizer import run_all_eda, plot_master_comparison, plot_confusion_matrix, plot_roc_curves, plot_training_curves
from src.model_baseline import BaselineModel
from src.model_ncf import NCFModel
from src.model_hybrid import HybridModel
from src.evaluator import Evaluator
from src.comparator import compare_all_models

def main():
    parser = argparse.ArgumentParser(description="Multi-Model E-Commerce Recommendation Engine")
    parser.add_argument('--skip-training', action='store_true', help="Skip training and load saved models")
    parser.add_argument('--user', type=str, default='U015', help="Target user for manual evaluation check")
    parser.add_argument('--top-k', type=int, default=5, help="Number of recommendations to generate")
    args = parser.parse_args()

    # 1. Data Preprocessing
    print("=" * 70)
    print("PHASE 1: DATA PREPROCESSING")
    print("=" * 70)
    preprocessor = DataPreprocessor(config.RAW_DATA_PATH)
    train_df, test_df, all_products_df = preprocessor.process()

    # 2. Exploratory Data Analysis (EDA)
    print("\n" + "=" * 70)
    print("PHASE 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    run_all_eda(train_df)

    # 3. Model Initialization
    evaluator = Evaluator(test_df, train_df, all_products_df)
    all_evaluation_results = []

    # =========================================================
    # MODEL 1: BASELINE (TF-IDF + KNN)
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 3: BASELINE MODEL (TF-IDF + KNN)")
    print("=" * 70)
    baseline_model = BaselineModel(train_df, all_products_df)
    baseline_model.build_tfidf()
    baseline_model.build_knn()
    
    base_results = evaluator.evaluate_model(
        model_name="Baseline (TF-IDF + KNN)",
        recommend_func=lambda user: baseline_model.recommend(user, top_k=args.top_k),
        top_k=args.top_k
    )
    evaluator.save_results(base_results, 'baseline_results.csv')
    all_evaluation_results.append(base_results)

    # =========================================================
    # MODEL 2: NCF (DEEP LEARNING)
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 4: DEEP LEARNING MODEL (NCF TWO-TOWER)")
    print("=" * 70)
    
    num_users = len(preprocessor.user_encoder.classes_)
    num_items = len(preprocessor.item_encoder.classes_)
    ncf_model = NCFModel(num_users, num_items)
    ncf_history = None
    
    if args.skip_training and os.path.exists(config.NCF_MODEL_PATH):
        ncf_model.load()
    else:
        ncf_model.build_model()
        ncf_history = ncf_model.train(
            train_df['user_encoded'].values,
            train_df['item_encoded'].values,
            train_df['rating'].values
        )
        ncf_model.save()

    # Plot training curves if history is available
    if ncf_history is not None:
        plot_training_curves(ncf_history)

    def ncf_recommend_wrapper(user):
        if user not in preprocessor.user_encoder.classes_:
            import pandas as pd
            return pd.DataFrame(columns=['productId', 'category']) # Cold start fallback
        user_enc = preprocessor.user_encoder.transform([user])[0]
        return ncf_model.recommend(user, user_enc, preprocessor.item_encoder, all_products_df, train_df, top_k=args.top_k)

    ncf_results = evaluator.evaluate_model(
        model_name="Deep Learning (NCF Two-Tower)",
        recommend_func=ncf_recommend_wrapper,
        top_k=args.top_k
    )
    evaluator.save_results(ncf_results, 'ncf_results.csv')
    all_evaluation_results.append(ncf_results)

    # =========================================================
    # MODEL 3: PROPOSED HYBRID (MINILM + SVD)
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 5: PROPOSED HYBRID MODEL (MiniLM + SVD)")
    print("=" * 70)
    hybrid_model = HybridModel(train_df, all_products_df)
    
    if args.skip_training and os.path.exists(config.EMBEDDINGS_PATH):
        hybrid_model.load_semantic_engine()
    else:
        hybrid_model.build_semantic_engine()
        
    if args.skip_training and os.path.exists(config.SVD_MODEL_PATH):
        hybrid_model.load_collaborative_engine()
    else:
        hybrid_model.build_collaborative_engine()

    hybrid_results = evaluator.evaluate_model(
        model_name="Proposed Hybrid (MiniLM + SVD)",
        recommend_func=lambda user: hybrid_model.recommend(user, top_k=args.top_k)[0],
        product_embeddings=hybrid_model.product_embeddings,
        top_k=args.top_k
    )
    evaluator.save_results(hybrid_results, 'hybrid_results.csv')
    all_evaluation_results.append(hybrid_results)

    # =========================================================
    # COMPARISON & REPORT
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 6: COMPARISON & REPORT")
    print("=" * 70)
    
    comparison_df = evaluator.generate_comparison_table(all_evaluation_results)
    plot_master_comparison(comparison_df)
    
    # NEW: Confusion Matrix & ROC Curves
    plot_confusion_matrix(all_evaluation_results)
    plot_roc_curves(all_evaluation_results)
    
    compare_all_models(
        target_user=args.user,
        train_df=train_df,
        all_products_df=all_products_df,
        baseline_model=baseline_model,
        ncf_model=ncf_model,
        hybrid_model=hybrid_model,
        user_encoder=preprocessor.user_encoder,
        item_encoder=preprocessor.item_encoder,
        top_k=args.top_k
    )
    
    print("\nPipeline execution completed successfully.")
    print(f"Check the {config.METRICS_DIR} and {config.PLOTS_DIR} directories for outputs.")

if __name__ == "__main__":
    main()
