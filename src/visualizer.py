import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc
from . import config

# Set global seaborn style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def plot_action_funnel(df):
    plt.figure(figsize=(10, 6))
    action_counts = df['action_type'].value_counts().reindex(['view', 'cart', 'purchase'])
    sns.barplot(y=action_counts.index, x=action_counts.values, palette="Blues_r")
    
    plt.title('Funnel Chart: Action Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Interactions', fontsize=12)
    plt.ylabel('Action Type', fontsize=12)
    
    for index, value in enumerate(action_counts.values):
        if not np.isnan(value):
            plt.text(value, index, f' {int(value)}', va='center', fontsize=12)
            
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, 'action_funnel.png'), dpi=300)
    plt.close()

def plot_interaction_trend(df):
    plt.figure(figsize=(14, 6))
    daily_interactions = df.groupby(df['timestamp'].dt.date).size()
    sns.lineplot(x=daily_interactions.index, y=daily_interactions.values, marker="o", color="coral", linewidth=2)
    
    plt.title('Interaction Trend Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Total Interactions', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, 'interaction_trend.png'), dpi=300)
    plt.close()

def plot_top_brands(df):
    plt.figure(figsize=(12, 6))
    top_brands = df['brand'].value_counts().head(10)
    sns.barplot(x=top_brands.values, y=top_brands.index, palette="viridis")
    
    plt.title('Top 10 Most Interacted Brands', fontsize=16, fontweight='bold')
    plt.xlabel('Interaction Count', fontsize=12)
    plt.ylabel('Brand', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, 'top_brands.png'), dpi=300)
    plt.close()

def plot_price_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['current_price'].dropna(), bins=30, kde=True, color="purple")
    
    plt.title('Product Price Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Current Price ($)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, 'price_distribution.png'), dpi=300)
    plt.close()

def plot_dynamic_weights():
    plt.figure(figsize=(10, 6))
    
    # Generate continuous t values
    t_days = np.linspace(0, 180, 500)
    alpha = config.DECAY_ALPHA_MIN + config.DECAY_ALPHA_RANGE * np.exp(-config.DECAY_LAMBDA * t_days)
    beta = 1.0 - alpha
    
    plt.plot(t_days, alpha, label='Semantic Weight (α)', color='#3498db', linewidth=2.5)
    plt.plot(t_days, beta, label='Collaborative Weight (β)', color='#9b59b6', linewidth=2.5)
    
    plt.title('Continuous Dynamic Adaptation of The Hybrid Brain', pad=20, fontweight='bold', fontsize=14)
    plt.xlabel('Days Since Last Interaction (t)', fontweight='bold')
    plt.ylabel('Weight Value', fontweight='bold')
    plt.axvline(x=7, color='r', linestyle='--', alpha=0.5, label='7-Day Mark')
    
    # Mark specific points
    points = [0, 14, 180]
    for p in points:
        a_val = config.DECAY_ALPHA_MIN + config.DECAY_ALPHA_RANGE * np.exp(-config.DECAY_LAMBDA * p)
        plt.scatter(p, a_val, color='#3498db', s=60, zorder=5)
        plt.scatter(p, 1-a_val, color='#9b59b6', s=60, zorder=5)
        
    plt.legend()
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, 'dynamic_weights_curve.png'), dpi=300)
    plt.close()


# =========================================================
# NEW VISUALIZATIONS
# =========================================================

def plot_confusion_matrix(all_results, save_dir=None):
    """
    Plot category-level confusion matrix for each model.
    Shows predicted category vs actual category heatmap.
    """
    if save_dir is None:
        save_dir = config.PLOTS_DIR
        
    for result in all_results:
        model_name = result['Model']
        pred_cats = result.get('_pred_categories', [])
        actual_cats = result.get('_actual_categories', [])
        
        if not pred_cats or not actual_cats:
            continue
        
        # Get unique labels
        all_labels = sorted(list(set(pred_cats + actual_cats)))
        
        cm = confusion_matrix(actual_cats, pred_cats, labels=all_labels)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=all_labels, yticklabels=all_labels,
                    linewidths=0.5, linecolor='gray')
        
        plt.title(f'Category Confusion Matrix\n{model_name}', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Predicted Category', fontsize=12, fontweight='bold')
        plt.ylabel('Actual Category', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        
        safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '').lower()
        plt.savefig(os.path.join(save_dir, f'confusion_matrix_{safe_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Confusion matrices saved.")


def plot_roc_curves(all_results, save_dir=None):
    """
    Plot ROC curves for all models on the same figure.
    Treats recommendation as binary classification (relevant vs not-relevant).
    """
    if save_dir is None:
        save_dir = config.PLOTS_DIR
    
    plt.figure(figsize=(10, 8))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    for idx, result in enumerate(all_results):
        model_name = result['Model']
        scores = result.get('_relevance_scores', [])
        labels = result.get('_relevance_labels', [])
        
        if not scores or not labels or sum(labels) == 0:
            continue
        
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        color = colors[idx % len(colors)]
        plt.plot(fpr, tpr, color=color, linewidth=2.5,
                 label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    # Plot diagonal (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1.5, label='Random (AUC = 0.5)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curve — Recommendation Relevance', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("ROC curves saved.")


def plot_training_curves(history, save_dir=None):
    """
    Plot NCF training loss and accuracy (MAE) curves.
    Creates two subplots: Loss curve and MAE curve with train vs validation.
    """
    if save_dir is None:
        save_dir = config.PLOTS_DIR
    
    if history is None:
        print("No training history available.")
        return
    
    hist = history.history if hasattr(history, 'history') else history
    epochs = range(1, len(hist['loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ---- Loss Curve ----
    ax1 = axes[0]
    ax1.plot(epochs, hist['loss'], 'o-', color='#e74c3c', linewidth=2, markersize=4, label='Training Loss')
    if 'val_loss' in hist:
        ax1.plot(epochs, hist['val_loss'], 's-', color='#3498db', linewidth=2, markersize=4, label='Validation Loss')
    
    ax1.set_title('NCF Training & Validation Loss', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Mark best validation loss
    if 'val_loss' in hist:
        best_epoch = np.argmin(hist['val_loss']) + 1
        best_val = min(hist['val_loss'])
        ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best @ Epoch {best_epoch}')
        ax1.scatter([best_epoch], [best_val], color='green', s=100, zorder=5, marker='*')
        ax1.annotate(f'Best: {best_val:.4f}', xy=(best_epoch, best_val), 
                     xytext=(best_epoch + 1, best_val + 0.1),
                     fontsize=10, fontweight='bold', color='green',
                     arrowprops=dict(arrowstyle='->', color='green'))
    
    # ---- Accuracy (MAE) Curve ----
    ax2 = axes[1]
    if 'mae' in hist:
        ax2.plot(epochs, hist['mae'], 'o-', color='#e74c3c', linewidth=2, markersize=4, label='Training MAE')
        if 'val_mae' in hist:
            ax2.plot(epochs, hist['val_mae'], 's-', color='#3498db', linewidth=2, markersize=4, label='Validation MAE')
        
        ax2.set_title('NCF Training & Validation MAE', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Mean Absolute Error', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Mark best validation MAE
        if 'val_mae' in hist:
            best_epoch_mae = np.argmin(hist['val_mae']) + 1
            best_val_mae = min(hist['val_mae'])
            ax2.axvline(x=best_epoch_mae, color='green', linestyle='--', alpha=0.7)
            ax2.scatter([best_epoch_mae], [best_val_mae], color='green', s=100, zorder=5, marker='*')
            ax2.annotate(f'Best: {best_val_mae:.4f}', xy=(best_epoch_mae, best_val_mae),
                         xytext=(best_epoch_mae + 1, best_val_mae + 0.05),
                         fontsize=10, fontweight='bold', color='green',
                         arrowprops=dict(arrowstyle='->', color='green'))
    else:
        ax2.text(0.5, 0.5, 'MAE metric not available', ha='center', va='center',
                 fontsize=14, transform=ax2.transAxes)
    
    plt.suptitle('NCF Deep Learning Model — Training Curves', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ncf_training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Training curves saved.")


def plot_master_comparison(results_df):
    if results_df is None or results_df.empty:
        return
        
    models = results_df['Model'].tolist()
    exact_precision = results_df['Exact_Match_Precision'].values * 100
    category_hit_rate = results_df['Category_Hit_Rate'].values * 100
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width/2, exact_precision, width, label='Exact Item Match (%)', color='#e74c3c')
    rects2 = ax.bar(x + width/2, category_hit_rate, width, label='Category Hit Rate (%)', color='#2ecc71')
    
    ax.set_ylabel('Success Rate / Percentage', fontweight='bold', fontsize=12)
    ax.set_title('Comprehensive Model Comparison on Sparse E-commerce Data', fontweight='bold', fontsize=14, pad=20)
    ax.set_xticks(x)
    
    # Create wrapped labels for better display
    wrapped_labels = [m.replace(" (", "\n(") for m in models]
    ax.set_xticklabels(wrapped_labels, fontweight='bold', fontsize=11)
    ax.legend(loc='upper right')
    ax.set_ylim(0, max(max(category_hit_rate), max(exact_precision)) + 20)
    
    ax.bar_label(rects1, padding=3, fmt='%.1f%%', fontweight='bold')
    ax.bar_label(rects2, padding=3, fmt='%.1f%%', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, 'master_comparison.png'), dpi=300)
    plt.close()


def run_all_eda(df):
    plot_action_funnel(df)
    plot_interaction_trend(df)
    plot_top_brands(df)
    plot_price_distribution(df)
    plot_dynamic_weights()
    print("EDA Visualizations saved.")
