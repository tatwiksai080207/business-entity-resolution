"""
Feature engineering module.
Extracts 20 lexical, phonetic, semantic, and structural similarity metrics
across candidate entity pairs.
"""

import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

NUM_REGEX = re.compile(r"\b\d+\b")


def compute_chunk_features(args):
    """Worker function for batch parallel feature computation."""
    start, end, s1_names, s2_names, s1_addrs, s2_addrs, s2_ids, tfidf_vals, emb_sims = args
    n = end - start

    name_fuzz = np.zeros(n, dtype=np.float32)
    name_partial = np.zeros(n, dtype=np.float32)
    name_tok_sort = np.zeros(n, dtype=np.float32)
    name_tok_set = np.zeros(n, dtype=np.float32)
    name_jw = np.zeros(n, dtype=np.float32)
    name_jacc3 = np.zeros(n, dtype=np.float32)
    name_tok_jacc = np.zeros(n, dtype=np.float32)
    name_l_ratio = np.zeros(n, dtype=np.float32)
    name_l_diff = np.zeros(n, dtype=np.float32)
    addr_fuzz = np.zeros(n, dtype=np.float32)
    addr_tok_sort = np.zeros(n, dtype=np.float32)
    addr_tok_set = np.zeros(n, dtype=np.float32)
    addr_tok_jacc = np.zeros(n, dtype=np.float32)
    addr_num_jacc = np.zeros(n, dtype=np.float32)
    addr_missing = np.zeros(n, dtype=np.float32)
    source_is_s3 = np.zeros(n, dtype=np.float32)
    combined = np.zeros(n, dtype=np.float32)
    high_conf = np.zeros(n, dtype=np.float32)

    for i in range(n):
        n1, n2 = s1_names[i], s2_names[i]
        a1, a2 = s1_addrs[i], s2_addrs[i]
        sid2 = s2_ids[i]

        if n1 and n2:
            r1 = fuzz.ratio(n1, n2) / 100.0
            r3 = fuzz.token_sort_ratio(n1, n2) / 100.0
            name_fuzz[i] = r1
            name_partial[i] = fuzz.partial_ratio(n1, n2) / 100.0
            name_tok_sort[i] = r3
            name_tok_set[i] = fuzz.token_set_ratio(n1, n2) / 100.0
            name_jw[i] = JaroWinkler.normalized_similarity(n1, n2)

            ng1 = set(n1[j : j + 3] for j in range(max(0, len(n1) - 2)))
            ng2 = set(n2[j : j + 3] for j in range(max(0, len(n2) - 2)))
            if ng1 or ng2:
                name_jacc3[i] = len(ng1 & ng2) / len(ng1 | ng2)

            t1, t2 = set(n1.split()), set(n2.split())
            if t1 or t2:
                name_tok_jacc[i] = len(t1 & t2) / len(t1 | t2)

            l1, l2 = len(n1), len(n2)
            name_l_ratio[i] = min(l1, l2) / max(l1, l2)
            name_l_diff[i] = abs(l1 - l2)
            if r1 > 0.95 or r3 > 0.95:
                high_conf[i] = 1.0

        if a1 and a2:
            addr_fuzz[i] = fuzz.ratio(a1, a2) / 100.0
            addr_tok_sort[i] = fuzz.token_sort_ratio(a1, a2) / 100.0
            addr_tok_set[i] = fuzz.token_set_ratio(a1, a2) / 100.0
            at1, at2 = set(a1.split()), set(a2.split())
            if at1 or at2:
                addr_tok_jacc[i] = len(at1 & at2) / len(at1 | at2)

            num1 = set(NUM_REGEX.findall(a1))
            num2 = set(NUM_REGEX.findall(a2))
            if num1 or num2:
                addr_num_jacc[i] = len(num1 & num2) / len(num1 | num2)
        else:
            addr_missing[i] = 1.0

        source_is_s3[i] = 1.0 if sid2.startswith("S3-") else 0.0
        ns = max(name_tok_sort[i], name_jw[i], tfidf_vals[i])
        as_val = max(addr_tok_sort[i], addr_fuzz[i]) if not addr_missing[i] else ns
        combined[i] = (2 * ns * as_val) / max(ns + as_val, 1e-4)

    return pd.DataFrame({
        "tfidf_cosine": tfidf_vals,
        "embed_cosine": emb_sims,
        "name_fuzz_ratio": name_fuzz,
        "name_partial": name_partial,
        "name_token_sort": name_tok_sort,
        "name_token_set": name_tok_set,
        "name_jaro_winkler": name_jw,
        "name_jaccard_3gram": name_jacc3,
        "name_token_jaccard": name_tok_jacc,
        "name_len_ratio": name_l_ratio,
        "name_len_diff": name_l_diff,
        "addr_fuzz_ratio": addr_fuzz,
        "addr_token_sort": addr_tok_sort,
        "addr_token_set": addr_tok_set,
        "addr_token_jaccard": addr_tok_jacc,
        "addr_numeric_jaccard": addr_num_jacc,
        "addr_missing": addr_missing,
        "source_is_s3": source_is_s3,
        "combined_score": combined,
        "high_conf_name": high_conf,
    })
