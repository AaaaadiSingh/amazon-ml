"""
Union + Top-K cap logic for combining multiple blockers.

Design: UNION all blocker outputs (never intersect — each blocker fails
differently). Then cap per Source-1 entity at a fixed top-K.

This module also emits candidate_pairs.tsv as a first-class pipeline artifact.
"""

import os
from typing import Dict, List, Optional

import pandas as pd


def union_candidates(
    *candidate_dicts: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """
    Union candidate sets from multiple blockers.

    Each input dict maps source1_id → list of candidate IDs.
    Returns the union, preserving order (first-seen order).
    """
    merged: Dict[str, List[str]] = {}

    for cand_dict in candidate_dicts:
        for s1_id, cand_list in cand_dict.items():
            if s1_id not in merged:
                merged[s1_id] = []
            seen = set(merged[s1_id])
            for cid in cand_list:
                if cid not in seen:
                    merged[s1_id].append(cid)
                    seen.add(cid)

    return merged


def cap_candidates(
    candidates: Dict[str, List[str]],
    top_k: int = 50,
) -> Dict[str, List[str]]:
    """
    Cap candidates per Source-1 entity at top-K.

    If we have ranking scores, they should be used to sort before capping.
    Currently just truncates to the first K (order from union).
    """
    return {
        s1_id: cand_list[:top_k]
        for s1_id, cand_list in candidates.items()
    }


def ensure_all_s1_covered(
    candidates: Dict[str, List[str]],
    all_s1_ids: List[str],
) -> Dict[str, List[str]]:
    """
    Ensure every Source 1 entity has an entry in the candidates dict,
    even if the candidate list is empty (valid for singletons).
    """
    for s1_id in all_s1_ids:
        if s1_id not in candidates:
            candidates[s1_id] = []
    return candidates


def emit_candidate_pairs_tsv(
    candidates: Dict[str, List[str]],
    output_path: str,
):
    """
    Write candidate_pairs.tsv — one row per Source-1 entity.

    Format:
        source1_entity_id<TAB>candidate_id_1;candidate_id_2;...

    Empty candidate lists produce a row with an empty second column.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    rows = []
    for s1_id in sorted(candidates.keys()):
        cand_str = ';'.join(candidates[s1_id])
        rows.append(f"{s1_id}\t{cand_str}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rows) + '\n')

    n_with = sum(1 for v in candidates.values() if v)
    n_without = sum(1 for v in candidates.values() if not v)
    total_pairs = sum(len(v) for v in candidates.values())
    print(f"  Wrote {output_path}: {len(candidates)} S1 entities, "
          f"{n_with} with candidates, {n_without} singletons, "
          f"{total_pairs} total pairs")


def measure_blocking_recall(
    candidates: Dict[str, List[str]],
    ground_truth: pd.DataFrame,
) -> dict:
    """
    Measure blocking recall — the fraction of true matches that appear in
    the candidate set. This is the CEILING on final F0.5.

    Parameters
    ----------
    candidates : dict
        source1_id → list of candidate IDs.
    ground_truth : pd.DataFrame
        Columns: source1_id, matching_ids (list of str).

    Returns
    -------
    dict
        'recall': float, 'total_true_pairs': int, 'recovered_pairs': int,
        'missed_pairs': list of (s1_id, missed_id).
    """
    total_true = 0
    recovered = 0
    missed_pairs = []

    for _, row in ground_truth.iterrows():
        s1_id = row['source1_id']
        true_matches = row['matching_ids']
        if not true_matches:
            continue

        cand_set = set(candidates.get(s1_id, []))
        for true_id in true_matches:
            total_true += 1
            if true_id in cand_set:
                recovered += 1
            else:
                missed_pairs.append((s1_id, true_id))

    recall = recovered / total_true if total_true > 0 else 1.0

    print(f"  Blocking recall: {recall:.4f} ({recovered}/{total_true})")
    if missed_pairs:
        print(f"  Missed {len(missed_pairs)} true pairs (showing first 10):")
        for s1_id, missed_id in missed_pairs[:10]:
            print(f"    {s1_id} → {missed_id}")

    return {
        'recall': recall,
        'total_true_pairs': total_true,
        'recovered_pairs': recovered,
        'missed_pairs': missed_pairs,
    }
