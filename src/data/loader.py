"""
Data loading utilities.

Handles reading Source 1/2/3 files and ground truth, with flexible format
detection (CSV/TSV) and consistent column naming.
"""

import os
import pandas as pd
from typing import Dict, Optional, Tuple


# Expected columns across sources
EXPECTED_COLUMNS = ['entity_id', 'business_name', 'business_address', 'country']


def detect_separator(filepath: str) -> str:
    """Auto-detect whether a file is tab-separated or comma-separated."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        first_line = f.readline()
    if '\t' in first_line:
        return '\t'
    return ','


def load_source_file(filepath: str, source_label: Optional[str] = None) -> pd.DataFrame:
    """
    Load a single source file (Source 1, 2, or 3).

    Parameters
    ----------
    filepath : str
        Path to the CSV/TSV file.
    source_label : str, optional
        Label to add as a 'source' column (e.g., 'S1', 'S2', 'S3').

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized columns.
    """
    sep = detect_separator(filepath)
    df = pd.read_csv(filepath, sep=sep, dtype=str, keep_default_na=False)

    # Normalize column names: lowercase, strip whitespace
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # Validate expected columns exist
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns {missing} in {filepath}. "
            f"Found columns: {list(df.columns)}"
        )

    if source_label:
        df['source'] = source_label

    return df[EXPECTED_COLUMNS + (['source'] if source_label else [])]


def load_ground_truth(filepath: str) -> pd.DataFrame:
    """
    Load ground truth file (train_ground_truth.tsv).

    Expected format: source_1_entity_id, matching_entity_ids (semicolon-separated or empty).

    Returns
    -------
    pd.DataFrame
        Columns: ['source1_id', 'matching_ids']
        matching_ids is a list of strings (possibly empty).
    """
    sep = detect_separator(filepath)
    df = pd.read_csv(filepath, sep=sep, dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # Try to identify columns by pattern
    # Common patterns: entity_id + matching_ids, or source1_id + matches
    if len(df.columns) >= 2:
        col_s1 = df.columns[0]
        col_matches = df.columns[1]
    else:
        raise ValueError(
            f"Ground truth file must have at least 2 columns, found: {list(df.columns)}"
        )

    result = pd.DataFrame()
    result['source1_id'] = df[col_s1].str.strip()
    result['matching_ids'] = df[col_matches].apply(
        lambda x: [mid.strip() for mid in x.split(';') if mid.strip()] if pd.notna(x) and x.strip() else []
    )

    return result


def load_all_sources(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load all source files from a directory.

    Looks for files matching patterns like source1.csv, source_1.tsv, etc.

    Parameters
    ----------
    data_dir : str
        Directory containing source files.

    Returns
    -------
    dict
        Keys: 'S1', 'S2', 'S3'. Values: DataFrames.
    """
    sources = {}
    patterns = {
        'S1': ['source1', 'source_1', 'src1'],
        'S2': ['source2', 'source_2', 'src2'],
        'S3': ['source3', 'source_3', 'src3'],
    }

    files_in_dir = os.listdir(data_dir)

    for label, name_patterns in patterns.items():
        found = False
        for fname in files_in_dir:
            fname_lower = fname.lower().replace('.csv', '').replace('.tsv', '')
            if any(p in fname_lower for p in name_patterns):
                filepath = os.path.join(data_dir, fname)
                sources[label] = load_source_file(filepath, source_label=label)
                print(f"  Loaded {label}: {filepath} ({len(sources[label])} records)")
                found = True
                break
        if not found:
            print(f"  WARNING: No file found for {label} in {data_dir}")

    return sources


def load_test_source1(data_dir: str) -> pd.DataFrame:
    """
    Load the test Source 1 file (the entities we need to match).

    Looks for files with 'test' in the name and 'source1' pattern.
    """
    files_in_dir = os.listdir(data_dir)
    for fname in files_in_dir:
        fname_lower = fname.lower()
        if 'test' in fname_lower and ('source1' in fname_lower or 'source_1' in fname_lower):
            filepath = os.path.join(data_dir, fname)
            return load_source_file(filepath, source_label='S1_test')

    raise FileNotFoundError(f"No test Source 1 file found in {data_dir}")


def create_train_val_split(
    ground_truth: pd.DataFrame,
    source1_df: pd.DataFrame,
    val_fraction: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ground truth into train/val, stratified by:
    - singleton vs. has-matches
    - country (from source1_df)

    Returns (train_gt, val_gt) DataFrames.
    """
    from sklearn.model_selection import train_test_split

    # Build stratification key
    gt = ground_truth.copy()
    gt['has_matches'] = gt['matching_ids'].apply(lambda x: len(x) > 0).astype(str)

    # Merge country from source1
    s1_country = source1_df[['entity_id', 'country']].copy()
    gt = gt.merge(s1_country, left_on='source1_id', right_on='entity_id', how='left')
    gt['country'] = gt['country'].fillna('unknown')

    # Stratification key
    gt['strat_key'] = gt['has_matches'] + '_' + gt['country']

    # Handle strata with too few samples
    strat_counts = gt['strat_key'].value_counts()
    small_strata = strat_counts[strat_counts < 2].index
    gt.loc[gt['strat_key'].isin(small_strata), 'strat_key'] = gt.loc[
        gt['strat_key'].isin(small_strata), 'has_matches'
    ]

    try:
        train_gt, val_gt = train_test_split(
            ground_truth,
            test_size=val_fraction,
            random_state=random_state,
            stratify=gt['strat_key'],
        )
    except ValueError:
        # Fallback: stratify only on has_matches
        train_gt, val_gt = train_test_split(
            ground_truth,
            test_size=val_fraction,
            random_state=random_state,
            stratify=gt['has_matches'],
        )

    print(f"  Train: {len(train_gt)} entities, Val: {len(val_gt)} entities")
    print(f"  Train singletons: {(train_gt['matching_ids'].apply(len) == 0).sum()}")
    print(f"  Val singletons: {(val_gt['matching_ids'].apply(len) == 0).sum()}")

    return train_gt, val_gt
