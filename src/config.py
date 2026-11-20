"""
Configuration module for Multilingual Business Entity Resolution.
Defines system paths, hardware detection, hyperparameters, and feature schemas.
"""

import os
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    GPU_COUNT = torch.cuda.device_count() if CUDA_AVAILABLE else 0
except ImportError:
    CUDA_AVAILABLE = False
    GPU_COUNT = 0

# Environment detection
IS_KAGGLE = os.path.exists("/kaggle")
IS_AWS = os.path.exists("/home/ubuntu")

if IS_KAGGLE:
    BASE_DIR = "/kaggle/working"
    DATA_DIR = "/kaggle/input/datasets/tatwiksaimolleti/amazon-ml-challenge-2026"
    TRAIN_DIR = os.path.join(DATA_DIR, "train") if os.path.exists(os.path.join(DATA_DIR, "train")) else DATA_DIR
    TEST_DIR = os.path.join(DATA_DIR, "test") if os.path.exists(os.path.join(DATA_DIR, "test")) else DATA_DIR
    OUTPUT_DIR = "/kaggle/working"
else:
    BASE_DIR = os.getcwd()
    TRAIN_DIR = os.path.join(BASE_DIR, "student_resource", "dataset", "train")
    TEST_DIR = os.path.join(BASE_DIR, "student_resource", "dataset", "test")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hardware settings
NUM_CPUS = os.cpu_count() or 4

# Dense embedding configuration
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384
EMBEDDING_BATCH_SIZE = 256 if IS_KAGGLE else 512

# Blocking configuration
MAX_CANDIDATES_PER_QUERY = 10
FAISS_TOP_K = 10
FAISS_SIMILARITY_THRESHOLD = 0.25
TFIDF_TOP_K = 8
TFIDF_SIMILARITY_THRESHOLD = 0.20

# Feature schema (20 dimensions)
FEATURE_COLUMNS = [
    "tfidf_cosine",
    "embed_cosine",
    "name_fuzz_ratio",
    "name_partial",
    "name_token_sort",
    "name_token_set",
    "name_jaro_winkler",
    "name_jaccard_3gram",
    "name_token_jaccard",
    "name_len_ratio",
    "name_len_diff",
    "addr_fuzz_ratio",
    "addr_token_sort",
    "addr_token_set",
    "addr_token_jaccard",
    "addr_numeric_jaccard",
    "addr_missing",
    "source_is_s3",
    "combined_score",
    "high_conf_name",
]

# Classification threshold
OPTIMAL_DECISION_THRESHOLD = 0.81
MAX_MATCHES_PER_ENTITY = 5
