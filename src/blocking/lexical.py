"""
Lexical blocker: character n-gram MinHash/LSH.

Catches typos and minor edits cheaply. Uses datasketch for MinHash + LSH.
"""

import re
from typing import Dict, List, Set, Tuple

import pandas as pd
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm


def char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from text."""
    if not text or len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def build_minhash(text: str, num_perm: int = 128, ngram_size: int = 3) -> MinHash:
    """Build a MinHash signature for a text string."""
    m = MinHash(num_perm=num_perm)
    for gram in char_ngrams(text, ngram_size):
        m.update(gram.encode('utf-8'))
    return m


class LexicalBlocker:
    """
    Character n-gram MinHash/LSH blocker.

    For each Source 1 entity, finds candidate matches in Source 2/3 based on
    approximate Jaccard similarity of character n-grams of the business name.
    """

    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.3,
        ngram_size: int = 3,
    ):
        """
        Parameters
        ----------
        num_perm : int
            Number of MinHash permutations (more = more accurate, slower).
        threshold : float
            LSH Jaccard threshold — lower catches more candidates (higher recall)
            but generates more pairs. Start low (0.3) since precision comes from the model.
        ngram_size : int
            Character n-gram size.
        """
        self.num_perm = num_perm
        self.threshold = threshold
        self.ngram_size = ngram_size
        self.lsh = None
        self.minhashes: Dict[str, MinHash] = {}

    def index(self, df: pd.DataFrame, text_col: str = 'name_normalized', id_col: str = 'entity_id'):
        """
        Build the LSH index over Source 2/3 records.

        Parameters
        ----------
        df : pd.DataFrame
            Source 2 and/or Source 3 records (already normalized).
        text_col : str
            Column containing the text to hash.
        id_col : str
            Column containing the entity ID.
        """
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self.minhashes = {}

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Lexical indexing"):
            eid = row[id_col]
            text = row[text_col]
            mh = build_minhash(text, self.num_perm, self.ngram_size)
            self.minhashes[eid] = mh
            try:
                self.lsh.insert(eid, mh)
            except ValueError:
                # Duplicate key — skip (shouldn't happen with unique entity_ids)
                pass

    def query(
        self,
        df_source1: pd.DataFrame,
        text_col: str = 'name_normalized',
        id_col: str = 'entity_id',
    ) -> Dict[str, List[str]]:
        """
        Query the LSH index for each Source 1 entity.

        Returns
        -------
        dict
            Mapping: source1_entity_id → list of candidate entity_ids from S2/S3.
        """
        if self.lsh is None:
            raise RuntimeError("Must call index() before query()")

        candidates = {}
        for _, row in tqdm(df_source1.iterrows(), total=len(df_source1), desc="Lexical blocking"):
            eid = row[id_col]
            text = row[text_col]
            mh = build_minhash(text, self.num_perm, self.ngram_size)
            results = self.lsh.query(mh)
            candidates[eid] = list(results)

        return candidates

    def query_with_name_and_address(
        self,
        df_source1: pd.DataFrame,
        df_targets: pd.DataFrame,
        name_col: str = 'name_normalized',
        addr_col: str = 'address_normalized',
        id_col: str = 'entity_id',
    ) -> Dict[str, List[str]]:
        """
        Run lexical blocking on concatenated name + address for better recall.
        Builds a fresh index on combined text.
        """
        # Build index on targets with combined text
        combined_lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)

        target_minhashes = {}
        for _, row in tqdm(df_targets.iterrows(), total=len(df_targets), desc="Lexical indexing (name+addr)"):
            eid = row[id_col]
            text = f"{row[name_col]} {row[addr_col]}"
            mh = build_minhash(text, self.num_perm, self.ngram_size)
            target_minhashes[eid] = mh
            try:
                combined_lsh.insert(eid, mh)
            except ValueError:
                pass

        # Query
        candidates = {}
        for _, row in tqdm(df_source1.iterrows(), total=len(df_source1), desc="Lexical blocking (name+addr)"):
            eid = row[id_col]
            text = f"{row[name_col]} {row[addr_col]}"
            mh = build_minhash(text, self.num_perm, self.ngram_size)
            results = combined_lsh.query(mh)
            candidates[eid] = list(results)

        return candidates
