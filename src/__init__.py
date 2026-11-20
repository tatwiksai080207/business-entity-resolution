"""
Multilingual Business Entity Resolution Package.
Amazon ML Challenge 2026.
"""

from .config import (
    FEATURE_COLUMNS,
    OPTIMAL_DECISION_THRESHOLD,
    MAX_CANDIDATES_PER_QUERY,
    MAX_MATCHES_PER_ENTITY,
)
from .preprocessing import normalize_name, normalize_address, preprocess_dataframe
from .blocking import tfidf_query_blocking, exact_name_blocking, faiss_dense_blocking
from .features import compute_chunk_features
from .model import train_lightgbm_model, sweep_f05_threshold, compute_entity_f05
from .inference import format_submission_files

__all__ = [
    "FEATURE_COLUMNS",
    "OPTIMAL_DECISION_THRESHOLD",
    "MAX_CANDIDATES_PER_QUERY",
    "MAX_MATCHES_PER_ENTITY",
    "normalize_name",
    "normalize_address",
    "preprocess_dataframe",
    "tfidf_query_blocking",
    "exact_name_blocking",
    "faiss_dense_blocking",
    "compute_chunk_features",
    "train_lightgbm_model",
    "sweep_f05_threshold",
    "compute_entity_f05",
    "format_submission_files",
]
