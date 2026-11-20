"""
Candidate generation and blocking module.
Combines dense semantic vector retrieval via FAISS GPU with sparse TF IDF n gram matching
and exact inverted index lookups.
"""

import time
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sparse_dot_topn import sp_matmul_topn
    USE_SPARSE_DOT_V1 = True
except ImportError:
    try:
        from sparse_dot_topn import awesome_cossim_topn
        USE_SPARSE_DOT_V1 = False
    except ImportError:
        USE_SPARSE_DOT_V1 = None

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


def run_sparse_topn(matrix_a, matrix_b_t, top_n, threshold, n_threads=4):
    """Wrapper for fast sparse matrix multiplication with top N filtering."""
    if USE_SPARSE_DOT_V1:
        return sp_matmul_topn(matrix_a, matrix_b_t, top_n=top_n, threshold=threshold, n_threads=n_threads)
    elif USE_SPARSE_DOT_V1 is False:
        return awesome_cossim_topn(
            matrix_a, matrix_b_t, ntop=top_n, lower_bound=threshold, use_threads=True, n_jobs=n_threads
        )
    else:
        raise ImportError("sparse_dot_topn library is required for fast sparse candidate blocking.")


def tfidf_query_blocking(s1_df, s2s3_df, field="name_norm", top_k=8, threshold=0.20, n_threads=4):
    """
    Chunked TF IDF candidate generation treating Source 1 as query rows.
    Fits n gram vocabulary strictly on Source 1 queries to avoid memory bloat.
    """
    n_s1 = len(s1_df)
    n_s2s3 = len(s2s3_df)
    if n_s1 == 0 or n_s2s3 == 0:
        return []

    s1_texts = s1_df[field].fillna("").tolist()
    s2s3_texts = s2s3_df[field].fillna("").tolist()
    s1_ids = s1_df["entity_id"].values
    s2s3_ids = s2s3_df["entity_id"].values

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 4),
        min_df=2,
        max_features=120000,
        dtype=np.float32,
        sublinear_tf=True,
    )
    s1_tfidf = vectorizer.fit_transform(s1_texts)

    chunk_size = 500000
    n_chunks = (n_s2s3 + chunk_size - 1) // chunk_size
    acc_s1, acc_s2s3, acc_score = [], [], []

    for ci in range(n_chunks):
        start = ci * chunk_size
        end = min(start + chunk_size, n_s2s3)
        chunk_tfidf = vectorizer.transform(s2s3_texts[start:end])
        cos_matrix = run_sparse_topn(s1_tfidf, chunk_tfidf.T, top_n=top_k, threshold=threshold, n_threads=n_threads)
        coo = cos_matrix.tocoo()

        if len(coo.data) > 0:
            acc_s1.append(coo.row.astype(np.int32))
            acc_s2s3.append((coo.col + start).astype(np.int32))
            acc_score.append(coo.data.astype(np.float32))

    if not acc_s1:
        return []

    cat_s1 = np.concatenate(acc_s1)
    cat_s2s3 = np.concatenate(acc_s2s3)
    cat_score = np.concatenate(acc_score)

    df_merge = pd.DataFrame({"s1": cat_s1, "s2s3": cat_s2s3, "sc": cat_score})
    df_merge = df_merge.sort_values("sc", ascending=False).groupby("s1").head(top_k).reset_index(drop=True)

    res_s1 = df_merge["s1"].values
    res_s2s3 = df_merge["s2s3"].values
    res_sc = df_merge["sc"].values

    return list(zip(s1_ids[res_s1], s2s3_ids[res_s2s3], res_sc.astype(float)))


def exact_name_blocking(s1_df, s2s3_df, max_per_query=5):
    """Instant exact match inverted index blocking for zero typo entity pairs."""
    s2s3_names = s2s3_df["name_norm"].values
    s2s3_ids = s2s3_df["entity_id"].values
    s1_names = s1_df["name_norm"].values
    s1_ids = s1_df["entity_id"].values

    name_index = defaultdict(list)
    for idx, name in enumerate(s2s3_names):
        if name and len(name) >= 3:
            name_index[name].append(s2s3_ids[idx])

    candidates = []
    for idx, name in enumerate(s1_names):
        if name and name in name_index:
            for mid in name_index[name][:max_per_query]:
                candidates.append((s1_ids[idx], mid, 1.0))

    return candidates


def faiss_dense_blocking(s1_df, s1_emb, s2s3_df, s2s3_emb, top_k=8, threshold=0.35, num_gpus=0):
    """
    Semantic nearest neighbor search over dense multilingual transformer representations
    using FAISS with GPU acceleration when available.
    """
    if not FAISS_AVAILABLE:
        return []

    n_s1 = len(s1_df)
    n_s2s3 = len(s2s3_df)
    if n_s1 == 0 or n_s2s3 == 0:
        return []

    s1_ids = s1_df["entity_id"].values
    s2s3_ids = s2s3_df["entity_id"].values
    dim = s1_emb.shape[1]

    index_cpu = faiss.IndexFlatIP(dim)
    index_cpu.add(s2s3_emb.astype(np.float32))

    if num_gpus > 0:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        except Exception:
            index = index_cpu
    else:
        index = index_cpu

    query_batch = 20000
    n_batches = (n_s1 + query_batch - 1) // query_batch
    candidates = []

    for qb in range(n_batches):
        q_start = qb * query_batch
        q_end = min(q_start + query_batch, n_s1)
        q_vecs = s1_emb[q_start:q_end]

        scores, indices = index.search(q_vecs, top_k)
        for i in range(len(q_vecs)):
            s1_idx = q_start + i
            for k in range(scores.shape[1]):
                sim = float(scores[i, k])
                target_idx = int(indices[i, k])
                if target_idx >= 0 and sim >= threshold:
                    candidates.append((s1_ids[s1_idx], s2s3_ids[target_idx], sim))

    return candidates
