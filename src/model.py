"""
LightGBM training, evaluation, and Macro F0.5 threshold optimization module.
"""

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit


def compute_entity_f05(y_true_set, y_pred_set):
    """
    Compute F0.5 score for a single Source 1 entity.
    Returns 1.0 for correctly identified singletons, 0.0 for false merges on singletons.
    """
    if len(y_true_set) == 0 and len(y_pred_set) == 0:
        return 1.0
    if len(y_true_set) == 0 and len(y_pred_set) > 0:
        return 0.0
    if len(y_true_set) > 0 and len(y_pred_set) == 0:
        return 0.0

    tp = len(y_true_set & y_pred_set)
    precision = tp / len(y_pred_set) if len(y_pred_set) > 0 else 0.0
    recall = tp / len(y_true_set) if len(y_true_set) > 0 else 0.0

    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    return (1.25 * precision * recall) / denom


def sweep_f05_threshold(val_df, s1_id_col="s1_id", s2_id_col="s2s3_id", prob_col="prob", gt_dict=None):
    """
    Vectorized threshold sweep to identify the exact probability cutoff
    maximizing Macro F0.5 across all validation entities.
    """
    thresholds = np.arange(0.50, 0.95, 0.02)
    best_thresh = 0.81
    best_score = -1.0

    all_val_s1 = list(gt_dict.keys()) if gt_dict else list(val_df[s1_id_col].unique())
    s1_candidates = val_df.groupby(s1_id_col)[[s2_id_col, prob_col]].apply(
        lambda g: list(zip(g[s2_id_col], g[prob_col]))
    ).to_dict()

    for thresh in thresholds:
        scores = []
        for s1 in all_val_s1:
            true_set = gt_dict.get(s1, set()) if gt_dict else set()
            cands = s1_candidates.get(s1, [])
            pred_set = {mid for mid, p in cands if p >= thresh}
            scores.append(compute_entity_f05(true_set, pred_set))

        macro_score = np.mean(scores)
        if macro_score > best_score:
            best_score = macro_score
            best_thresh = thresh

    return best_thresh, best_score


def train_lightgbm_model(x_train, y_train, groups_train=None, x_val=None, y_val=None):
    """Train LightGBM binary classifier with precision oriented objective."""
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "n_jobs": -1,
    }

    train_data = lgb.Dataset(x_train, label=y_train)
    val_data = lgb.Dataset(x_val, label=y_val, reference=train_data) if x_val is not None else None

    valid_sets = [train_data]
    if val_data:
        valid_sets.append(val_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=valid_sets,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    return model
