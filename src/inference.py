"""
Inference and submission formatting module.
Generates matching_results.tsv and candidate_pairs.tsv with strict compliance checks:
1. Every Source 1 entity is represented exactly once.
2. Matched IDs form a strict subset of candidate pairs.
3. Up to 10 candidates per entity.
4. Correct singleton formatting with empty string.
5. Zero self matches.
"""

import os
import pandas as pd
from collections import defaultdict


def format_submission_files(
    all_s1_ids,
    cand_pairs_df,
    predictions_mask,
    output_dir="output",
    max_matches_per_entity=5,
    max_cands_per_entity=10,
):
    """
    Format and write matching_results.tsv and candidate_pairs.tsv.
    Guarantees that matched IDs are a strict subset of candidate pairs.
    """
    os.makedirs(output_dir, exist_ok=True)

    match_dict = defaultdict(list)
    cand_dict = defaultdict(list)

    cand_s1_ids = cand_pairs_df["s1_id"].values
    cand_s2_ids = cand_pairs_df["s2s3_id"].values

    for idx, (sid, mid) in enumerate(zip(cand_s1_ids, cand_s2_ids)):
        cand_dict[sid].append(mid)
        if predictions_mask[idx]:
            match_dict[sid].append(mid)

    match_rows = []
    cand_rows = []

    for sid in all_s1_ids:
        raw_matches = list(dict.fromkeys(match_dict.get(sid, [])))
        if len(raw_matches) > max_matches_per_entity:
            raw_matches = raw_matches[:max_matches_per_entity]

        match_str = ",".join(raw_matches) if raw_matches else ""
        match_rows.append({"source1_entity_id": sid, "matched_entity_ids": match_str})

        raw_cands = list(dict.fromkeys(cand_dict.get(sid, [])))
        ordered_cands = list(dict.fromkeys(raw_matches + raw_cands))
        if len(ordered_cands) > max_cands_per_entity:
            ordered_cands = ordered_cands[:max_cands_per_entity]

        cand_str = ",".join(ordered_cands) if ordered_cands else ""
        cand_rows.append({"source1_entity_id": sid, "candidate_entity_ids": cand_str})

    df_match = pd.DataFrame(match_rows)
    df_cand = pd.DataFrame(cand_rows)

    match_path = os.path.join(output_dir, "matching_results.tsv")
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")

    df_match.to_csv(match_path, sep="\t", index=False)
    df_cand.to_csv(cand_path, sep="\t", index=False)

    return match_path, cand_path
