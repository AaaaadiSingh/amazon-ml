"""
Custom per-entity macro-averaged F0.5 evaluation.

CRITICAL: This is NOT sklearn's fbeta_score called on flattened predictions.
The challenge metric is:
1. Compute F0.5 PER Source-1 entity
2. Macro-average across all Source-1 entities
3. A correctly predicted empty list for a true singleton scores 1.0
4. A wrong match on a true singleton scores 0.0

This custom implementation is the ONLY valid local evaluation — never use
sklearn's global f1_score/fbeta_score as a substitute.
"""

from typing import Dict, List, Set, Optional, Tuple
import numpy as np


def f05_score(precision: float, recall: float) -> float:
    """
    Compute F0.5 from precision and recall.

    F_beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
    For beta = 0.5: (1 + 0.25) * P * R / (0.25 * P + R) = 1.25 * P * R / (0.25P + R)
    """
    beta_sq = 0.25  # beta=0.5 → beta^2=0.25
    if precision + recall == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def compute_entity_f05(
    predicted_matches: List[str],
    true_matches: List[str],
) -> float:
    """
    Compute F0.5 for a single Source-1 entity.

    Special cases:
    - True singleton (no matches) + predicted empty → 1.0 (correct!)
    - True singleton + predicted non-empty → 0.0 (false merge, catastrophic)
    - True has matches + predicted empty → 0.0 (missed all)
    - True has matches + predicted matches → standard F0.5
    """
    pred_set = set(predicted_matches) if predicted_matches else set()
    true_set = set(true_matches) if true_matches else set()

    # Case: true singleton
    if len(true_set) == 0:
        if len(pred_set) == 0:
            return 1.0  # Correctly predicted empty
        else:
            return 0.0  # False merge on a singleton — worst case

    # Case: has true matches
    if len(pred_set) == 0:
        return 0.0  # Missed everything

    # Standard precision/recall/F0.5
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return f05_score(precision, recall)


def compute_macro_f05(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
) -> float:
    """
    Compute macro-averaged F0.5 across all Source-1 entities.

    Parameters
    ----------
    predictions : dict
        s1_id → list of predicted matching entity IDs.
    ground_truth : dict
        s1_id → set of true matching entity IDs.

    Returns
    -------
    float
        Macro-averaged F0.5 score.
    """
    all_s1_ids = set(list(predictions.keys()) + list(ground_truth.keys()))

    if not all_s1_ids:
        return 0.0

    entity_scores = []
    for s1_id in all_s1_ids:
        pred = predictions.get(s1_id, [])
        true = list(ground_truth.get(s1_id, set()))
        entity_scores.append(compute_entity_f05(pred, true))

    return float(np.mean(entity_scores))


def evaluate_predictions(
    predictions: Dict[str, List[str]],
    ground_truth_df,
    verbose: bool = True,
) -> Dict:
    """
    Full evaluation with detailed breakdown.

    Parameters
    ----------
    predictions : dict
        s1_id → list of predicted matching entity IDs.
    ground_truth_df : pd.DataFrame
        Columns: source1_id, matching_ids (list of str).
    verbose : bool
        Print detailed results.

    Returns
    -------
    dict
        'macro_f05': float,
        'per_entity_scores': dict,
        'breakdown': dict with category-level stats.
    """
    # Build GT lookup
    gt_lookup = {}
    for _, row in ground_truth_df.iterrows():
        gt_lookup[row['source1_id']] = set(row['matching_ids'])

    # Compute per-entity scores
    per_entity = {}
    categories = {
        'singleton_correct': [],     # True singleton, predicted empty
        'singleton_wrong': [],       # True singleton, predicted non-empty
        'has_match_correct': [],     # Has matches, got some right
        'has_match_missed': [],      # Has matches, predicted empty
        'has_match_partial': [],     # Has matches, partially correct
    }

    all_s1_ids = set(list(predictions.keys()) + list(gt_lookup.keys()))

    for s1_id in all_s1_ids:
        pred = predictions.get(s1_id, [])
        true = gt_lookup.get(s1_id, set())

        score = compute_entity_f05(pred, list(true))
        per_entity[s1_id] = {
            'score': score,
            'predicted': pred,
            'true': list(true),
        }

        # Categorize
        if len(true) == 0:
            if len(pred) == 0:
                categories['singleton_correct'].append(s1_id)
            else:
                categories['singleton_wrong'].append(s1_id)
        else:
            if len(pred) == 0:
                categories['has_match_missed'].append(s1_id)
            elif score == 1.0:
                categories['has_match_correct'].append(s1_id)
            else:
                categories['has_match_partial'].append(s1_id)

    macro_f05 = float(np.mean([v['score'] for v in per_entity.values()]))

    if verbose:
        print(f"\n{'='*60}")
        print(f"  MACRO F0.5: {macro_f05:.4f}")
        print(f"{'='*60}")
        print(f"  Total entities: {len(all_s1_ids)}")
        print(f"  Singletons correct (predicted empty): {len(categories['singleton_correct'])}")
        print(f"  Singletons WRONG (false merge!):      {len(categories['singleton_wrong'])}")
        print(f"  Has-match correct (perfect F0.5=1.0): {len(categories['has_match_correct'])}")
        print(f"  Has-match missed (predicted empty):   {len(categories['has_match_missed'])}")
        print(f"  Has-match partial:                    {len(categories['has_match_partial'])}")

        # Show worst false merges on singletons
        if categories['singleton_wrong']:
            print(f"\n  CRITICAL: {len(categories['singleton_wrong'])} false merges on singletons!")
            print(f"  (These score 0.0 each — fixing these is the highest-ROI improvement)")
            for s1_id in categories['singleton_wrong'][:5]:
                info = per_entity[s1_id]
                print(f"    {s1_id}: predicted {info['predicted']}")

    return {
        'macro_f05': macro_f05,
        'per_entity_scores': per_entity,
        'breakdown': {k: len(v) for k, v in categories.items()},
        'categories': categories,
    }
