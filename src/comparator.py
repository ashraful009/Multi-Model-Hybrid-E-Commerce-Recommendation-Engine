import pandas as pd
import numpy as np

def compare_all_models(target_user, train_df, all_products_df, baseline_model, ncf_model, hybrid_model, user_encoder, item_encoder, top_k=5):
    print("\n" + "=" * 70)
    print(f"🔍 ANALYZING USER ID: {target_user}")
    print("=" * 70)

    # Determine User Type & Status
    global_max_date = train_df['timestamp'].max()
    user_history = train_df[train_df['userId'] == target_user].sort_values(by='timestamp', ascending=False)

    if user_history.empty:
        user_type = "🆕 NEW USER (Cold Start)"
        status = "No previous history found."
    else:
        user_last_seen = user_history['timestamp'].max()
        days_since_active = (global_max_date - user_last_seen).total_seconds() / 86400.0

        if days_since_active > 7:
            user_type = "🕰️ LONG-TERM USER (Historical/Inactive)"
            status = f"Last active {days_since_active:.1f} days ago."
        else:
            user_type = "⚡ SHORT-TERM USER (Active Shopper)"
            status = f"Recently active ({days_since_active:.1f} days ago)."

    print(f"👤 USER TYPE: {user_type}")
    print(f"📊 STATUS:    {status}")

    if not user_history.empty:
        print("\n📜 RECENT HISTORY (Last 3 Interactions):")
        recent = user_history[['productId', 'product_name', 'category', 'action_type']].head(3)
        for _, row in recent.iterrows():
            print(f"  [{row['action_type'].upper()}] {row['productId']} | {row['product_name'][:40]}... | {row['category']}")

    print("-" * 70)

    # 1. Baseline Model
    print("\n🤖 MODEL 1: Baseline (TF-IDF + KNN) -> The 'Filter Bubble' Model")
    try:
        base_recs = baseline_model.recommend(target_user, top_k=top_k)
        if base_recs.empty:
            print("⚠️ Not enough data for baseline.")
        else:
            for i, row in base_recs.iterrows():
                print(f"  {i+1}. {row['productId']} | {row['product_name'][:40]}... | {row['category']} (Score: {row['Final_Score']:.4f})")
    except Exception as e:
        print(f"⚠️ Baseline Model Error: {e}")

    # 2. NCF Model
    print("\n🧠 MODEL 2: Deep Learning (NCF Two-Tower) -> The 'Broad Category' Model")
    try:
        if target_user not in user_encoder.classes_:
            print("⚠️ Cold Start: NCF cannot predict for completely new users without retraining.")
        else:
            user_enc = user_encoder.transform([target_user])[0]
            ncf_recs = ncf_model.recommend(target_user, user_enc, item_encoder, all_products_df, train_df, top_k=top_k)
            for i, row in ncf_recs.iterrows():
                print(f"  {i+1}. {row['productId']} | {row['product_name'][:40]}... | {row['category']}")
    except Exception as e:
        print(f"⚠️ NCF Model Error: {e}")

    # 3. Hybrid Model
    print("\n👑 MODEL 3: Proposed Hybrid Model -> The 'Smart Cross-Selling' Winner")
    try:
        hybrid_recs, t_days, alpha, beta = hybrid_model.recommend(target_user, top_k=top_k)
        print(f"   [Dynamic Weights: t={t_days:.1f} days, α(Semantic)={alpha:.2f}, β(Collab)={beta:.2f}]")
        if hybrid_recs.empty:
            print("⚠️ No recommendations returned.")
        else:
            for i, row in hybrid_recs.iterrows():
                print(f"  {i+1}. {row['productId']} | {row['product_name'][:40]}... | {row['category']} (Score: {row['Final_Score']:.4f})")
    except Exception as e:
        print(f"⚠️ Hybrid Model Error: {e}")

    print("=" * 70)
