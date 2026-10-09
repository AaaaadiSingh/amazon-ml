"""
Pairwise feature engineering for entity resolution.

Computes similarity features between a Source 1 record and a candidate
Source 2/3 record. All features are designed to be robust to unseen countries
(France) — no country-specific logic.

Feature groups:
- Name similarity (Jaro-Winkler, token-sort, token-set, n-gram cosine, phonetic)
- Address similarity (token Jaccard, number match, postal match)
- Meta (country match flag, source pair, length deltas)
- Embedding cosine (reused from blocking, if available)
"""

import re
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

try:
    import jellyfish
except ImportError:
    jellyfish = None


# ── Name similarity features ────────────────────────────────────────────────

def jaro_winkler_sim(s1: str, s2: str) -> float:
    """Jaro-Winkler similarity (0-1)."""
    if not s1 or not s2:
        return 0.0
    return JaroWinkler.normalized_similarity(s1, s2)


def token_sort_ratio(s1: str, s2: str) -> float:
    """Token-sort ratio from rapidfuzz (0-100, normalized to 0-1)."""
    if not s1 or not s2:
        return 0.0
    return fuzz.token_sort_ratio(s1, s2) / 100.0


def token_set_ratio(s1: str, s2: str) -> float:
    """Token-set ratio from rapidfuzz (0-100, normalized to 0-1)."""
    if not s1 or not s2:
        return 0.0
    return fuzz.token_set_ratio(s1, s2) / 100.0


def partial_ratio(s1: str, s2: str) -> float:
    """Partial string ratio (good for substring matches, 0-1)."""
    if not s1 or not s2:
        return 0.0
    return fuzz.partial_ratio(s1, s2) / 100.0


def wratio(s1: str, s2: str) -> float:
    """Weighted ratio combining multiple fuzzy strategies (0-1)."""
    if not s1 or not s2:
        return 0.0
    return fuzz.WRatio(s1, s2) / 100.0


def char_ngram_overlap(s1: str, s2: str, n: int = 3) -> float:
    """
    Character n-gram Jaccard overlap coefficient.
    """
    if not s1 or not s2:
        return 0.0
    grams1 = {s1[i:i+n] for i in range(max(0, len(s1) - n + 1))}
    grams2 = {s2[i:i+n] for i in range(max(0, len(s2) - n + 1))}
    if not grams1 or not grams2:
        return 0.0
    intersection = grams1 & grams2
    union = grams1 | grams2
    return len(intersection) / len(union)


def phonetic_match(s1: str, s2: str) -> float:
    """
    Phonetic similarity using Double Metaphone.
    Returns 1.0 if primary codes match, 0.5 if secondary matches, 0.0 otherwise.
    """
    if jellyfish is None or not s1 or not s2:
        return 0.0
    try:
        # Compare metaphone codes of first significant word
        words1 = s1.split()
        words2 = s2.split()
        if not words1 or not words2:
            return 0.0
        code1 = jellyfish.metaphone(words1[0])
        code2 = jellyfish.metaphone(words2[0])
        if code1 and code2 and code1 == code2:
            return 1.0
    except Exception:
        pass
    return 0.0


def exact_match_after_norm(s1: str, s2: str) -> float:
    """Exact match on normalized text (1.0 or 0.0)."""
    if not s1 or not s2:
        return 0.0
    return 1.0 if s1 == s2 else 0.0


def name_containment(s1: str, s2: str) -> float:
    """
    How much of the shorter name's tokens are contained in the longer name.
    Useful for catching 'ABC Corp' ⊂ 'ABC Corporation Private Limited'.
    """
    if not s1 or not s2:
        return 0.0
    tokens1 = set(s1.split())
    tokens2 = set(s2.split())
    if not tokens1 or not tokens2:
        return 0.0
    shorter, longer = (tokens1, tokens2) if len(tokens1) <= len(tokens2) else (tokens2, tokens1)
    return len(shorter & longer) / len(shorter)


# ── Address similarity features ─────────────────────────────────────────────

def token_jaccard(s1: str, s2: str) -> float:
    """Token-level Jaccard similarity."""
    if not s1 or not s2:
        return 0.0
    tokens1 = set(s1.split())
    tokens2 = set(s2.split())
    if not tokens1 or not tokens2:
        return 0.0
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)


def number_tokens_match(s1: str, s2: str) -> float:
    """Fraction of numeric tokens in s1 that also appear in s2."""
    nums1 = set(re.findall(r'\b\d+\b', s1)) if s1 else set()
    nums2 = set(re.findall(r'\b\d+\b', s2)) if s2 else set()
    if not nums1:
        return 0.0
    return len(nums1 & nums2) / len(nums1)


def postal_code_match(s1: str, s2: str) -> float:
    """Whether postal/PIN codes match (5-6 digit sequences)."""
    postals1 = set(re.findall(r'\b\d{5,6}\b', s1)) if s1 else set()
    postals2 = set(re.findall(r'\b\d{5,6}\b', s2)) if s2 else set()
    if not postals1 or not postals2:
        return 0.0
    return 1.0 if postals1 & postals2 else 0.0


# ── Meta features ───────────────────────────────────────────────────────────

def country_match(c1: str, c2: str) -> float:
    """Soft country match flag (1.0 if same, 0.0 if different)."""
    if not c1 or not c2:
        return 0.0
    return 1.0 if c1.lower().strip() == c2.lower().strip() else 0.0


def length_ratio(s1: str, s2: str) -> float:
    """Length ratio of two strings (always ≤ 1.0)."""
    if not s1 or not s2:
        return 0.0
    l1, l2 = len(s1), len(s2)
    if max(l1, l2) == 0:
        return 0.0
    return min(l1, l2) / max(l1, l2)


def token_count_diff(s1: str, s2: str) -> int:
    """Absolute difference in token counts."""
    t1 = len(s1.split()) if s1 else 0
    t2 = len(s2.split()) if s2 else 0
    return abs(t1 - t2)


# ── Feature vector construction ─────────────────────────────────────────────

# Ordered list of feature names (used as column names in the feature matrix)
FEATURE_NAMES = [
    # Name features
    'name_jaro_winkler',
    'name_token_sort_ratio',
    'name_token_set_ratio',
    'name_partial_ratio',
    'name_wratio',
    'name_char_3gram_jaccard',
    'name_phonetic_match',
    'name_exact_match',
    'name_containment',
    'name_length_ratio',
    'name_token_count_diff',
    # Name on raw (for typo distance)
    'name_raw_jaro_winkler',
    'name_raw_token_sort_ratio',
    # Address features
    'addr_token_jaccard',
    'addr_jaro_winkler',
    'addr_token_sort_ratio',
    'addr_number_match',
    'addr_postal_match',
    'addr_length_ratio',
    'addr_token_count_diff',
    # Landmark
    'landmark_token_jaccard',
    # Meta
    'country_match',
    'name_len_s1',
    'name_len_s2',
    'addr_len_s1',
    'addr_len_s2',
]


def compute_pairwise_features(
    s1_record: dict,
    s2_record: dict,
    embedding_cosine: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute all pairwise features between a Source 1 and Source 2/3 record.

    Parameters
    ----------
    s1_record : dict
        Source 1 record with both raw and normalized fields.
    s2_record : dict
        Source 2/3 record with both raw and normalized fields.
    embedding_cosine : float, optional
        Pre-computed embedding cosine similarity (from the embedding blocker).

    Returns
    -------
    dict
        Feature name → feature value.
    """
    name1_norm = s1_record.get('name_normalized', '')
    name2_norm = s2_record.get('name_normalized', '')
    name1_raw = s1_record.get('business_name', '').lower()
    name2_raw = s2_record.get('business_name', '').lower()
    addr1 = s1_record.get('address_normalized', '')
    addr2 = s2_record.get('address_normalized', '')
    lm1 = s1_record.get('landmark', '')
    lm2 = s2_record.get('landmark', '')
    c1 = s1_record.get('country_normalized', '')
    c2 = s2_record.get('country_normalized', '')

    features = {
        # Name similarity (normalized)
        'name_jaro_winkler': jaro_winkler_sim(name1_norm, name2_norm),
        'name_token_sort_ratio': token_sort_ratio(name1_norm, name2_norm),
        'name_token_set_ratio': token_set_ratio(name1_norm, name2_norm),
        'name_partial_ratio': partial_ratio(name1_norm, name2_norm),
        'name_wratio': wratio(name1_norm, name2_norm),
        'name_char_3gram_jaccard': char_ngram_overlap(name1_norm, name2_norm, 3),
        'name_phonetic_match': phonetic_match(name1_norm, name2_norm),
        'name_exact_match': exact_match_after_norm(name1_norm, name2_norm),
        'name_containment': name_containment(name1_norm, name2_norm),
        'name_length_ratio': length_ratio(name1_norm, name2_norm),
        'name_token_count_diff': float(token_count_diff(name1_norm, name2_norm)),

        # Name similarity (raw — captures typo distance)
        'name_raw_jaro_winkler': jaro_winkler_sim(name1_raw, name2_raw),
        'name_raw_token_sort_ratio': token_sort_ratio(name1_raw, name2_raw),

        # Address similarity
        'addr_token_jaccard': token_jaccard(addr1, addr2),
        'addr_jaro_winkler': jaro_winkler_sim(addr1, addr2),
        'addr_token_sort_ratio': token_sort_ratio(addr1, addr2),
        'addr_number_match': number_tokens_match(addr1, addr2),
        'addr_postal_match': postal_code_match(addr1, addr2),
        'addr_length_ratio': length_ratio(addr1, addr2),
        'addr_token_count_diff': float(token_count_diff(addr1, addr2)),

        # Landmark
        'landmark_token_jaccard': token_jaccard(lm1, lm2) if (lm1 and lm2) else 0.0,

        # Meta
        'country_match': country_match(c1, c2),
        'name_len_s1': float(len(name1_norm)),
        'name_len_s2': float(len(name2_norm)),
        'addr_len_s1': float(len(addr1)),
        'addr_len_s2': float(len(addr2)),
    }

    # Embedding cosine (if provided)
    if embedding_cosine is not None:
        features['embedding_cosine'] = embedding_cosine

    return features


def build_feature_matrix(
    candidates: Dict[str, List[str]],
    s1_lookup: Dict[str, dict],
    s23_lookup: Dict[str, dict],
    embedding_cosines: Optional[Dict[Tuple[str, str], float]] = None,
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """
    Build the full feature matrix for all candidate pairs.

    Parameters
    ----------
    candidates : dict
        source1_id → list of candidate IDs.
    s1_lookup : dict
        entity_id → record dict for Source 1.
    s23_lookup : dict
        entity_id → record dict for Source 2/3.
    embedding_cosines : dict, optional
        (s1_id, s2_id) → cosine similarity.

    Returns
    -------
    (feature_df, pair_ids)
        feature_df: DataFrame with one row per candidate pair, columns = feature names.
        pair_ids: list of (source1_id, candidate_id) tuples in same order.
    """
    from tqdm import tqdm

    all_features = []
    pair_ids = []

    total_pairs = sum(len(cands) for cands in candidates.values())

    with tqdm(total=total_pairs, desc="Computing features") as pbar:
        for s1_id, cand_list in candidates.items():
            s1_rec = s1_lookup.get(s1_id, {})
            for cand_id in cand_list:
                s2_rec = s23_lookup.get(cand_id, {})

                emb_cos = None
                if embedding_cosines and (s1_id, cand_id) in embedding_cosines:
                    emb_cos = embedding_cosines[(s1_id, cand_id)]

                feats = compute_pairwise_features(s1_rec, s2_rec, emb_cos)
                all_features.append(feats)
                pair_ids.append((s1_id, cand_id))
                pbar.update(1)

    feature_df = pd.DataFrame(all_features)
    print(f"  Feature matrix: {feature_df.shape[0]} pairs × {feature_df.shape[1]} features")

    return feature_df, pair_ids


def build_labels(
    pair_ids: List[Tuple[str, str]],
    ground_truth: pd.DataFrame,
) -> np.ndarray:
    """
    Build binary label array for candidate pairs.

    Parameters
    ----------
    pair_ids : list of (s1_id, candidate_id)
    ground_truth : pd.DataFrame
        Columns: source1_id, matching_ids (list of str).

    Returns
    -------
    np.ndarray
        Binary labels: 1 = true match, 0 = non-match.
    """
    # Build lookup: s1_id → set of true match IDs
    gt_lookup = {}
    for _, row in ground_truth.iterrows():
        gt_lookup[row['source1_id']] = set(row['matching_ids'])

    labels = np.zeros(len(pair_ids), dtype=np.float32)
    for i, (s1_id, cand_id) in enumerate(pair_ids):
        if cand_id in gt_lookup.get(s1_id, set()):
            labels[i] = 1.0

    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    print(f"  Labels: {n_pos} positives, {n_neg} negatives "
          f"(ratio: 1:{n_neg/max(n_pos,1):.1f})")

    return labels
