import os

# Project Roots
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_DIR = os.path.join(ROOT_DIR, 'dataset')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'evaluation_plots')
METRICS_DIR = os.path.join(RESULTS_DIR, 'evaluation_results')

# Paths
RAW_DATA_PATH = os.path.join(DATASET_DIR, 'ecommarce_full_dataset.csv')
TRAIN_DATA_PATH = os.path.join(DATASET_DIR, 'phase1_train_data.csv')
TEST_DATA_PATH = os.path.join(DATASET_DIR, 'phase1_test_data.csv')

SVD_MODEL_PATH = os.path.join(MODELS_DIR, 'svd_model.pkl')
NCF_MODEL_PATH = os.path.join(MODELS_DIR, 'ncf_model.h5')
EMBEDDINGS_PATH = os.path.join(MODELS_DIR, 'product_embeddings.pkl')
USER_ENCODER_PATH = os.path.join(MODELS_DIR, 'user_encoder.pkl')
ITEM_ENCODER_PATH = os.path.join(MODELS_DIR, 'item_encoder.pkl')

# Hyperparameters
TF_IDF_MAX_FEATURES = 5000
KNN_K = 30
NCF_EMBEDDING_DIM = 64
NCF_EPOCHS = 20
NCF_BATCH_SIZE = 128
NCF_LR = 0.001
NCF_NEGATIVE_SAMPLES = 4
SVD_N_FACTORS = 100
SVD_N_EPOCHS = 20
SVD_LR = 0.005
SVD_REG = 0.02
MINILM_MODEL_NAME = 'all-MiniLM-L6-v2'
TOP_K = 5
HYBRID_RECENT_ITEMS = 5

# Decay Parameters for Continuous Weighting
# Formula: alpha = DECAY_ALPHA_MIN + DECAY_ALPHA_RANGE * exp(-DECAY_LAMBDA * t)
DECAY_ALPHA_MIN = 0.3
DECAY_ALPHA_RANGE = 0.4
DECAY_LAMBDA = 0.1

# Temporal Split Date
SPLIT_DATE = '2025-12-01'
