"""
End-to-end pipeline for business entity resolution.

Orchestrates: load → normalize → block → features → train/predict → decide → output.

Usage:
    # Training + validation
    python -m src.pipeline --mode train --data-dir data/raw --output-dir output

    # Prediction on test data
    python -m src.pipeline --mode predict --data-dir data/raw --output-dir output --model-dir models

    # Quick baseline (rule-based, no ML model)
    python -m src.pipeline --mode baseline --data-dir data/raw --output-dir output
"""

import argparse
import os
import sys
import time
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import (
    load_all_sources,
    load_ground_truth,
    load_test_source1,
    create_train_val_split,
)
from src.data.normalize import normalize_dataframe
from src.blocking.lexical import LexicalBlocker
from src.blocking.address import AddressBlocker
from src.blocking.union import (
    union_candidates,
    cap_candidates,
    ensure_all_s1_covered,
    emit_candidate_pairs_tsv,
    measure_blocking_recall,
)
from src.features.pairwise import (
    build_feature_matrix,
    build_labels,
    FEATURE_NAMES,
)
from src.models.xgboost_scorer import XGBoostScorer
from src.decision.calibration import DecisionLayer
from src.collective.consistency import resolve_conflicts
from src.eval.metrics import evaluate_predictions


def df_to_lookup(df: pd.DataFrame) -> Dict[str, dict]:
    """Convert a DataFrame to a dict of {entity_id: row_dict}."""
    return {row['entity_id']: row.to_dict() for _, row in df.iterrows()}


def emit_matching_results_tsv(
    predictions: Dict[str, List[str]],
    output_path: str,
):
    """
    Write matching_results.tsv — one row per Source-1 entity.

    Format:
        source1_entity_id<TAB>match_id_1;match_id_2;...

    Empty match lists produce a row with an empty second column.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    rows = []
    for s1_id in sorted(predictions.keys()):
        match_str = ';'.join(predictions[s1_id])
        rows.append(f"{s1_id}\t{match_str}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rows) + '\n')

    n_with = sum(1 for v in predictions.values() if v)
    n_empty = sum(1 for v in predictions.values() if not v)
    print(f"  Wrote {output_path}: {len(predictions)} entities, "
          f"{n_with} with matches, {n_empty} singletons")


def run_blocking(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    top_k: int = 50,
    use_embeddings: bool = False,
    embedding_cache_dir: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Run all blockers and union results.

    Parameters
    ----------
    s1_df : pd.DataFrame
        Normalized Source 1 records.
    s23_df : pd.DataFrame
        Normalized Source 2+3 records.
    top_k : int
        Max candidates per Source 1 entity.
    use_embeddings : bool
        Whether to use the embedding blocker (requires sentence-transformers + FAISS).
    embedding_cache_dir : str, optional
        Directory to cache embeddings for re-use.

    Returns
    -------
    dict
        s1_id → list of candidate entity IDs.
    """
    all_candidates = []

    # 1. Lexical blocker (char n-gram MinHash/LSH on name)
    print("\n── Lexical Blocking (name) ──")
    lex_blocker = LexicalBlocker(num_perm=128, threshold=0.3, ngram_size=3)
    lex_blocker.index(s23_df, text_col='name_normalized')
    lex_cands = lex_blocker.query(s1_df, text_col='name_normalized')
    total_lex = sum(len(v) for v in lex_cands.values())
    print(f"  Lexical (name): {total_lex} total candidate pairs")
    all_candidates.append(lex_cands)

    # 1b. Lexical blocker on name+address (broader recall)
    print("\n── Lexical Blocking (name+address) ──")
    lex_blocker2 = LexicalBlocker(num_perm=128, threshold=0.25, ngram_size=3)
    lex_cands2 = lex_blocker2.query_with_name_and_address(s1_df, s23_df)
    total_lex2 = sum(len(v) for v in lex_cands2.values())
    print(f"  Lexical (name+addr): {total_lex2} total candidate pairs")
    all_candidates.append(lex_cands2)

    # 2. Address blocker (token overlap)
    print("\n── Address Blocking ──")
    addr_blocker = AddressBlocker(min_token_overlap=2, postal_match_boost=True)
    addr_blocker.index(s23_df, addr_col='address_normalized')
    addr_cands = addr_blocker.query(s1_df, addr_col='address_normalized')
    total_addr = sum(len(v) for v in addr_cands.values())
    print(f"  Address: {total_addr} total candidate pairs")
    all_candidates.append(addr_cands)

    # 3. Embedding blocker (optional — needs GPU for initial encoding)
    if use_embeddings:
        print("\n── Embedding Blocking ──")
        try:
            from src.blocking.embedding import EmbeddingBlocker
            emb_blocker = EmbeddingBlocker(top_k=20)

            target_cache = os.path.join(embedding_cache_dir, 'target_embeddings.pkl') if embedding_cache_dir else None
            s1_cache = os.path.join(embedding_cache_dir, 's1_embeddings.pkl') if embedding_cache_dir else None

            emb_blocker.encode_targets(s23_df, cache_path=target_cache)
            emb_cands = emb_blocker.query(s1_df, cache_path=s1_cache)
            total_emb = sum(len(v) for v in emb_cands.values())
            print(f"  Embedding: {total_emb} total candidate pairs")
            all_candidates.append(emb_cands)
        except Exception as e:
            print(f"  WARNING: Embedding blocker failed: {e}")
            print(f"  Continuing without embedding blocking")

    # Union and cap
    print("\n── Union + Cap ──")
    unified = union_candidates(*all_candidates)
    unified = cap_candidates(unified, top_k=top_k)
    unified = ensure_all_s1_covered(unified, s1_df['entity_id'].tolist())

    total_final = sum(len(v) for v in unified.values())
    print(f"  Final candidates: {total_final} pairs across {len(unified)} S1 entities")

    return unified


def run_baseline(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    ground_truth: Optional[pd.DataFrame] = None,
    output_dir: str = 'output',
):
    """
    Quick rule-based baseline: high-similarity threshold on name matching.

    Gets a valid submission on the board fast (Day 1, Hours 2-6 of the plan).
    No ML model — just fuzzy string matching with a conservative threshold.
    """
    from rapidfuzz import fuzz as rfuzz

    print("\n" + "="*60)
    print("  BASELINE: Rule-based name matching")
    print("="*60)

    s1_lookup = df_to_lookup(s1_df)
    s23_lookup = df_to_lookup(s23_df)

    # Simple blocking: for each S1 entity, compare against all S2/S3
    # This is O(N*M) but fine for small datasets
    predictions = {}
    candidates_dict = {}

    s23_ids = s23_df['entity_id'].tolist()

    for _, s1_row in s1_df.iterrows():
        s1_id = s1_row['entity_id']
        s1_name = s1_row['name_normalized']

        scored_pairs = []
        for s23_id in s23_ids:
            s23_row = s23_lookup[s23_id]
            s23_name = s23_row['name_normalized']

            # Quick filter: skip if no character overlap at all
            score = rfuzz.token_set_ratio(s1_name, s23_name) / 100.0
            if score > 0.5:
                scored_pairs.append((s23_id, score))

        # Sort by score
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        # Candidates: top 20 by score
        candidates_dict[s1_id] = [pair[0] for pair in scored_pairs[:20]]

        # Matches: only very high confidence (precision-first)
        matches = [pair[0] for pair in scored_pairs if pair[1] >= 0.85]
        predictions[s1_id] = matches

    # Emit outputs
    os.makedirs(output_dir, exist_ok=True)
    emit_candidate_pairs_tsv(candidates_dict, os.path.join(output_dir, 'candidate_pairs.tsv'))
    emit_matching_results_tsv(predictions, os.path.join(output_dir, 'matching_results.tsv'))

    # Evaluate if ground truth available
    if ground_truth is not None:
        print("\n── Baseline Evaluation ──")
        results = evaluate_predictions(predictions, ground_truth)
        return results

    return None


def run_train(
    data_dir: str,
    output_dir: str = 'output',
    model_dir: str = 'models',
    use_embeddings: bool = False,
    top_k: int = 50,
    val_fraction: float = 0.2,
):
    """
    Full training pipeline:
    1. Load and normalize data
    2. Split into train/val
    3. Run blocking
    4. Build features
    5. Train XGBoost
    6. Calibrate and find optimal threshold
    7. Evaluate on validation
    8. Save model + produce output files
    """
    start_time = time.time()

    print("\n" + "="*60)
    print("  FULL TRAINING PIPELINE")
    print("="*60)

    # ── Step 1: Load data ──
    print("\n── Step 1: Loading data ──")
    sources = load_all_sources(data_dir)
    if 'S1' not in sources:
        raise FileNotFoundError("Source 1 not found. Check your data directory.")

    s1_df = sources['S1']
    s23_dfs = [sources[k] for k in ['S2', 'S3'] if k in sources]
    if not s23_dfs:
        raise FileNotFoundError("No Source 2 or Source 3 found.")
    s23_df = pd.concat(s23_dfs, ignore_index=True)

    # Load ground truth
    gt_files = [f for f in os.listdir(data_dir) if 'ground_truth' in f.lower() or 'gt' in f.lower()]
    if not gt_files:
        raise FileNotFoundError("No ground truth file found in data directory.")
    gt_path = os.path.join(data_dir, gt_files[0])
    ground_truth = load_ground_truth(gt_path)
    print(f"  Ground truth: {len(ground_truth)} entities")

    # ── Step 2: Normalize ──
    print("\n── Step 2: Normalizing ──")
    s1_df = normalize_dataframe(s1_df)
    s23_df = normalize_dataframe(s23_df)
    print(f"  S1: {len(s1_df)} records, S2+S3: {len(s23_df)} records")

    # ── Step 3: Train/val split ──
    print("\n── Step 3: Train/val split ──")
    train_gt, val_gt = create_train_val_split(ground_truth, s1_df, val_fraction=val_fraction)

    # Split S1 accordingly
    train_s1_ids = set(train_gt['source1_id'])
    val_s1_ids = set(val_gt['source1_id'])
    s1_train = s1_df[s1_df['entity_id'].isin(train_s1_ids)]
    s1_val = s1_df[s1_df['entity_id'].isin(val_s1_ids)]

    # ── Step 4: Blocking ──
    print("\n── Step 4: Blocking (train set) ──")
    embedding_cache = os.path.join(model_dir, 'embedding_cache') if use_embeddings else None
    train_candidates = run_blocking(s1_train, s23_df, top_k=top_k,
                                     use_embeddings=use_embeddings,
                                     embedding_cache_dir=embedding_cache)

    print("\n── Blocking recall (train) ──")
    measure_blocking_recall(train_candidates, train_gt)

    print("\n── Step 4b: Blocking (val set) ──")
    val_candidates = run_blocking(s1_val, s23_df, top_k=top_k,
                                   use_embeddings=use_embeddings,
                                   embedding_cache_dir=embedding_cache)

    print("\n── Blocking recall (val) ──")
    measure_blocking_recall(val_candidates, val_gt)

    # ── Step 5: Feature engineering ──
    print("\n── Step 5: Feature engineering ──")
    s1_lookup = df_to_lookup(s1_df)
    s23_lookup = df_to_lookup(s23_df)

    X_train, train_pair_ids = build_feature_matrix(train_candidates, s1_lookup, s23_lookup)
    y_train = build_labels(train_pair_ids, train_gt)

    X_val, val_pair_ids = build_feature_matrix(val_candidates, s1_lookup, s23_lookup)
    y_val = build_labels(val_pair_ids, val_gt)

    # ── Step 6: Train XGBoost ──
    print("\n── Step 6: Training XGBoost ──")
    scorer = XGBoostScorer()
    train_result = scorer.train(X_train, y_train, X_val, y_val)

    # ── Step 7: Calibrate + threshold ──
    print("\n── Step 7: Calibration + threshold optimization ──")
    val_raw_scores = scorer.predict(X_val)

    decision = DecisionLayer()
    decision.fit(val_raw_scores, y_val, val_pair_ids, val_gt)

    # ── Step 8: Evaluate on validation ──
    print("\n── Step 8: Validation evaluation ──")
    val_predictions = decision.predict(
        val_raw_scores, val_pair_ids, list(val_s1_ids)
    )

    # Collective consistency
    score_lookup = {(s1_id, cand_id): score
                    for (s1_id, cand_id), score in zip(val_pair_ids, val_raw_scores)}
    val_predictions = resolve_conflicts(val_predictions, score_lookup)

    val_results = evaluate_predictions(val_predictions, val_gt)

    # ── Step 9: Save model ──
    print("\n── Step 9: Saving model ──")
    os.makedirs(model_dir, exist_ok=True)
    scorer.save(os.path.join(model_dir, 'xgboost_scorer'))

    with open(os.path.join(model_dir, 'decision_layer.pkl'), 'wb') as f:
        pickle.dump(decision, f)
    print(f"  Decision layer saved to {model_dir}/decision_layer.pkl")

    # ── Step 10: Generate full predictions (for submission) ──
    print("\n── Step 10: Generating full predictions ──")
    full_candidates = run_blocking(s1_df, s23_df, top_k=top_k,
                                    use_embeddings=use_embeddings,
                                    embedding_cache_dir=embedding_cache)

    X_full, full_pair_ids = build_feature_matrix(full_candidates, s1_lookup, s23_lookup)
    full_raw_scores = scorer.predict(X_full)
    full_predictions = decision.predict(
        full_raw_scores, full_pair_ids, s1_df['entity_id'].tolist()
    )
    full_predictions = resolve_conflicts(
        full_predictions,
        {(s1, c): s for (s1, c), s in zip(full_pair_ids, full_raw_scores)}
    )

    # Emit output files
    os.makedirs(output_dir, exist_ok=True)
    emit_candidate_pairs_tsv(full_candidates, os.path.join(output_dir, 'candidate_pairs.tsv'))
    emit_matching_results_tsv(full_predictions, os.path.join(output_dir, 'matching_results.tsv'))

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed/60:.1f} minutes")
    print(f"  Validation macro F0.5: {val_results['macro_f05']:.4f}")
    print(f"  Output files in: {output_dir}/")
    print(f"{'='*60}")

    return val_results


def run_predict(
    data_dir: str,
    output_dir: str = 'output',
    model_dir: str = 'models',
    use_embeddings: bool = False,
    top_k: int = 50,
):
    """
    Prediction-only pipeline for test data.
    Loads a trained model and generates matching_results.tsv + candidate_pairs.tsv.
    """
    print("\n" + "="*60)
    print("  PREDICTION PIPELINE (TEST)")
    print("="*60)

    # Load test Source 1
    print("\n── Loading test data ──")
    try:
        s1_test = load_test_source1(data_dir)
    except FileNotFoundError:
        # Fallback: use regular source files
        sources = load_all_sources(data_dir)
        s1_test = sources.get('S1')
        if s1_test is None:
            raise FileNotFoundError("No test Source 1 file found.")

    s23_dfs = []
    sources = load_all_sources(data_dir)
    for k in ['S2', 'S3']:
        if k in sources:
            s23_dfs.append(sources[k])
    s23_df = pd.concat(s23_dfs, ignore_index=True)

    # Normalize
    s1_test = normalize_dataframe(s1_test)
    s23_df = normalize_dataframe(s23_df)

    # Load model
    scorer = XGBoostScorer()
    scorer.load(os.path.join(model_dir, 'xgboost_scorer'))

    with open(os.path.join(model_dir, 'decision_layer.pkl'), 'rb') as f:
        decision = pickle.load(f)

    # Block → features → predict → decide
    embedding_cache = os.path.join(model_dir, 'embedding_cache') if use_embeddings else None
    candidates = run_blocking(s1_test, s23_df, top_k=top_k,
                               use_embeddings=use_embeddings,
                               embedding_cache_dir=embedding_cache)

    s1_lookup = df_to_lookup(s1_test)
    s23_lookup = df_to_lookup(s23_df)
    X, pair_ids = build_feature_matrix(candidates, s1_lookup, s23_lookup)

    raw_scores = scorer.predict(X)
    predictions = decision.predict(raw_scores, pair_ids, s1_test['entity_id'].tolist())

    score_lookup = {(s1, c): s for (s1, c), s in zip(pair_ids, raw_scores)}
    predictions = resolve_conflicts(predictions, score_lookup)

    # Emit
    os.makedirs(output_dir, exist_ok=True)
    emit_candidate_pairs_tsv(candidates, os.path.join(output_dir, 'candidate_pairs.tsv'))
    emit_matching_results_tsv(predictions, os.path.join(output_dir, 'matching_results.tsv'))

    print(f"\n  Done. Output in: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Business Entity Resolution Pipeline')
    parser.add_argument('--mode', choices=['train', 'predict', 'baseline'],
                        default='train', help='Pipeline mode')
    parser.add_argument('--data-dir', default='data/raw', help='Directory with source data files')
    parser.add_argument('--output-dir', default='output', help='Directory for output TSV files')
    parser.add_argument('--model-dir', default='models', help='Directory for model artifacts')
    parser.add_argument('--use-embeddings', action='store_true',
                        help='Use embedding blocker (requires GPU for first run)')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Max candidates per Source 1 entity')
    parser.add_argument('--val-fraction', type=float, default=0.2,
                        help='Validation split fraction')

    args = parser.parse_args()

    if args.mode == 'baseline':
        # Quick baseline — load, normalize, run simple matcher
        sources = load_all_sources(args.data_dir)
        s1_df = normalize_dataframe(sources['S1'])
        s23_dfs = [normalize_dataframe(sources[k]) for k in ['S2', 'S3'] if k in sources]
        s23_df = pd.concat(s23_dfs, ignore_index=True)

        gt_files = [f for f in os.listdir(args.data_dir) if 'ground_truth' in f.lower()]
        gt = load_ground_truth(os.path.join(args.data_dir, gt_files[0])) if gt_files else None

        run_baseline(s1_df, s23_df, ground_truth=gt, output_dir=args.output_dir)

    elif args.mode == 'train':
        run_train(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            use_embeddings=args.use_embeddings,
            top_k=args.top_k,
            val_fraction=args.val_fraction,
        )

    elif args.mode == 'predict':
        run_predict(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            use_embeddings=args.use_embeddings,
            top_k=args.top_k,
        )


if __name__ == '__main__':
    main()
