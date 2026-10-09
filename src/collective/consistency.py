"""
Collective consistency pass — conflict resolution across clusters.

Cheap version (Section 4.6 of the design doc):
If the same S2/S3 ID is a confident match for more than one S1 entity,
keep only the highest-scoring assignment and drop the rest.

A single S2/S3 record shouldn't legitimately belong to two different
real businesses, so this is a low-risk precision fix.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


def resolve_conflicts(
    predictions: Dict[str, List[str]],
    scores: Optional[Dict[Tuple[str, str], float]] = None,
) -> Dict[str, List[str]]:
    """
    Resolve conflicts where the same S2/S3 entity is matched to multiple S1 entities.

    If scores are available, keeps the highest-scoring assignment.
    If not, keeps the first-seen assignment (arbitrary but deterministic).

    Parameters
    ----------
    predictions : dict
        s1_id → list of matched entity IDs.
    scores : dict, optional
        (s1_id, matched_id) → score. Used to break ties.

    Returns
    -------
    dict
        Cleaned predictions with conflicts resolved.
    """
    # Build reverse map: matched_id → list of (s1_id, score)
    reverse_map: Dict[str, List[Tuple[str, float]]] = {}

    for s1_id, matched_ids in predictions.items():
        for mid in matched_ids:
            score = scores.get((s1_id, mid), 0.0) if scores else 0.0
            if mid not in reverse_map:
                reverse_map[mid] = []
            reverse_map[mid].append((s1_id, score))

    # Find conflicts
    conflicts_resolved = 0
    ids_to_remove: Dict[str, set] = {}  # s1_id → set of match IDs to remove

    for mid, assignments in reverse_map.items():
        if len(assignments) <= 1:
            continue

        # Conflict: same matched_id assigned to multiple S1 entities
        # Keep only the highest-scoring assignment
        assignments.sort(key=lambda x: x[1], reverse=True)
        winner = assignments[0][0]

        for s1_id, _ in assignments[1:]:
            if s1_id not in ids_to_remove:
                ids_to_remove[s1_id] = set()
            ids_to_remove[s1_id].add(mid)
            conflicts_resolved += 1

    # Apply removals
    cleaned = {}
    for s1_id, matched_ids in predictions.items():
        remove_set = ids_to_remove.get(s1_id, set())
        cleaned[s1_id] = [mid for mid in matched_ids if mid not in remove_set]

    if conflicts_resolved > 0:
        print(f"  Collective consistency: resolved {conflicts_resolved} conflicts")
    else:
        print(f"  Collective consistency: no conflicts found")

    return cleaned
