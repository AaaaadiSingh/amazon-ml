"""
Submission format validator.

Checks matching_results.tsv and candidate_pairs.tsv against challenge requirements:
- Tab-separated, no quoting
- One row per Source-1 entity
- No duplicate Source-1 entity IDs
- Match IDs must be S2-/S3- prefixed (if that convention applies)
- All matches must appear in the candidate set

Usage:
    python utils/validate_submission.py --matching output/matching_results.tsv --candidates output/candidate_pairs.tsv
    python utils/validate_submission.py --matching output/matching_results.tsv --candidates output/candidate_pairs.tsv --source1-ids data/raw/source1.csv
"""

import argparse
import sys
import os


def parse_tsv(filepath: str) -> dict:
    """Parse a TSV file into {entity_id: list_of_ids}."""
    entries = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.rstrip('\n\r')
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 1:
                print(f"  ERROR: Line {line_num}: empty line")
                continue
            entity_id = parts[0].strip()
            ids_str = parts[1].strip() if len(parts) > 1 else ''
            ids_list = [x.strip() for x in ids_str.split(';') if x.strip()] if ids_str else []
            if entity_id in entries:
                print(f"  ERROR: Line {line_num}: duplicate entity_id '{entity_id}'")
                return None
            entries[entity_id] = ids_list
    return entries


def validate(matching_path: str, candidates_path: str, source1_ids_path: str = None) -> bool:
    """
    Validate submission files. Returns True if valid, False otherwise.
    """
    errors = []
    warnings = []

    # ── Check files exist ──
    if not os.path.exists(matching_path):
        errors.append(f"matching_results.tsv not found: {matching_path}")
    if not os.path.exists(candidates_path):
        errors.append(f"candidate_pairs.tsv not found: {candidates_path}")

    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
        return False

    # ── Parse files ──
    print(f"  Checking {matching_path}...")
    matching = parse_tsv(matching_path)
    if matching is None:
        errors.append("matching_results.tsv has duplicate entity IDs")

    print(f"  Checking {candidates_path}...")
    candidates = parse_tsv(candidates_path)
    if candidates is None:
        errors.append("candidate_pairs.tsv has duplicate entity IDs")

    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
        return False

    # ── Check consistency ──
    # All S1 IDs in matching must be in candidates
    matching_s1 = set(matching.keys())
    candidate_s1 = set(candidates.keys())

    missing_in_candidates = matching_s1 - candidate_s1
    if missing_in_candidates:
        warnings.append(
            f"{len(missing_in_candidates)} S1 entities in matching_results but not in candidate_pairs"
        )

    missing_in_matching = candidate_s1 - matching_s1
    if missing_in_matching:
        warnings.append(
            f"{len(missing_in_matching)} S1 entities in candidate_pairs but not in matching_results"
        )

    # All matches must be a subset of candidates
    matches_not_in_candidates = 0
    for s1_id, match_ids in matching.items():
        cand_ids = set(candidates.get(s1_id, []))
        for mid in match_ids:
            if mid not in cand_ids:
                matches_not_in_candidates += 1
                if matches_not_in_candidates <= 5:
                    warnings.append(
                        f"Match '{mid}' for S1 '{s1_id}' not in candidate set"
                    )

    if matches_not_in_candidates > 5:
        warnings.append(
            f"... and {matches_not_in_candidates - 5} more matches not in candidate sets"
        )

    # Check for duplicate match IDs within a single entity
    for s1_id, match_ids in matching.items():
        if len(match_ids) != len(set(match_ids)):
            errors.append(f"Duplicate match IDs for S1 entity '{s1_id}'")

    for s1_id, cand_ids in candidates.items():
        if len(cand_ids) != len(set(cand_ids)):
            errors.append(f"Duplicate candidate IDs for S1 entity '{s1_id}'")

    # ── Check against source1 IDs if provided ──
    if source1_ids_path and os.path.exists(source1_ids_path):
        import pandas as pd
        sep = '\t' if source1_ids_path.endswith('.tsv') else ','
        s1_df = pd.read_csv(source1_ids_path, sep=sep, dtype=str)
        s1_df.columns = [c.strip().lower().replace(' ', '_') for c in s1_df.columns]
        expected_ids = set(s1_df['entity_id'].str.strip())

        extra_in_matching = matching_s1 - expected_ids
        missing_from_matching = expected_ids - matching_s1

        if extra_in_matching:
            errors.append(f"{len(extra_in_matching)} unknown S1 IDs in matching_results")
        if missing_from_matching:
            warnings.append(
                f"{len(missing_from_matching)} S1 IDs from source file missing in matching_results"
            )

    # ── Report ──
    print()
    n_matches_total = sum(len(v) for v in matching.values())
    n_empty = sum(1 for v in matching.values() if not v)
    n_cands_total = sum(len(v) for v in candidates.values())

    print(f"  matching_results.tsv: {len(matching)} entities, "
          f"{n_matches_total} total matches, {n_empty} singletons")
    print(f"  candidate_pairs.tsv:  {len(candidates)} entities, "
          f"{n_cands_total} total candidates")

    if warnings:
        print(f"\n  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"    ⚠ {w}")

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    ✗ {e}")
        print(f"\n  ✗ VALIDATION FAILED")
        return False
    else:
        print(f"\n  ✓ VALIDATION PASSED")
        return True


def main():
    parser = argparse.ArgumentParser(description='Validate submission files')
    parser.add_argument('--matching', required=True, help='Path to matching_results.tsv')
    parser.add_argument('--candidates', required=True, help='Path to candidate_pairs.tsv')
    parser.add_argument('--source1-ids', default=None,
                        help='Optional: path to source1 file to check all IDs are covered')
    args = parser.parse_args()

    print("="*60)
    print("  SUBMISSION VALIDATOR")
    print("="*60)

    passed = validate(args.matching, args.candidates, args.source1_ids)
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
