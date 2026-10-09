"""
Address blocker: token overlap on normalized addresses.

Catches matches where the business name differs but they share the same
physical location (street, city, PIN/postal code).
"""

import re
from typing import Dict, List, Set

import pandas as pd
from tqdm import tqdm


def tokenize_address(address: str) -> Set[str]:
    """
    Tokenize an address into a set of meaningful tokens.
    Filters out very short tokens (1 char) to reduce noise.
    """
    if not address:
        return set()
    tokens = address.lower().split()
    return {t for t in tokens if len(t) > 1}


def extract_postal_tokens(address: str) -> Set[str]:
    """Extract tokens that look like postal/PIN codes (5-6 digit sequences)."""
    return set(re.findall(r'\b\d{5,6}\b', address))


def extract_number_tokens(address: str) -> Set[str]:
    """Extract all numeric tokens from an address."""
    return set(re.findall(r'\b\d+\b', address))


class AddressBlocker:
    """
    Address-based blocker using token overlap.

    Generates candidates based on shared address tokens:
    - City/locality words
    - Street numbers
    - PIN/postal codes (strongest signal)
    """

    def __init__(
        self,
        min_token_overlap: int = 2,
        postal_match_boost: bool = True,
        max_candidates_per_query: int = 50,
    ):
        """
        Parameters
        ----------
        min_token_overlap : int
            Minimum number of shared address tokens to consider a candidate.
        postal_match_boost : bool
            If True, any postal code match automatically qualifies as a candidate
            regardless of other token overlap.
        max_candidates_per_query : int
            Cap on candidates per Source 1 entity from this blocker alone.
        """
        self.min_token_overlap = min_token_overlap
        self.postal_match_boost = postal_match_boost
        self.max_candidates_per_query = max_candidates_per_query
        self.target_tokens: Dict[str, Set[str]] = {}
        self.target_postals: Dict[str, Set[str]] = {}
        # Inverted index: token → set of entity IDs
        self.token_to_entities: Dict[str, Set[str]] = {}
        self.postal_to_entities: Dict[str, Set[str]] = {}

    def index(
        self,
        df_targets: pd.DataFrame,
        addr_col: str = 'address_normalized',
        id_col: str = 'entity_id',
    ):
        """
        Build an inverted index over Source 2/3 address tokens.
        """
        self.target_tokens = {}
        self.target_postals = {}
        self.token_to_entities = {}
        self.postal_to_entities = {}

        for _, row in tqdm(df_targets.iterrows(), total=len(df_targets), desc="Address indexing"):
            eid = row[id_col]
            addr = str(row[addr_col])

            tokens = tokenize_address(addr)
            postals = extract_postal_tokens(addr)

            self.target_tokens[eid] = tokens
            self.target_postals[eid] = postals

            for token in tokens:
                if token not in self.token_to_entities:
                    self.token_to_entities[token] = set()
                self.token_to_entities[token].add(eid)

            for postal in postals:
                if postal not in self.postal_to_entities:
                    self.postal_to_entities[postal] = set()
                self.postal_to_entities[postal].add(eid)

        print(f"  Address index: {len(self.target_tokens)} entities, "
              f"{len(self.token_to_entities)} unique tokens, "
              f"{len(self.postal_to_entities)} unique postal codes")

    def query(
        self,
        df_source1: pd.DataFrame,
        addr_col: str = 'address_normalized',
        id_col: str = 'entity_id',
    ) -> Dict[str, List[str]]:
        """
        Find address-based candidates for each Source 1 entity.

        Uses inverted index for efficiency: for each S1 entity, look up which
        S2/S3 entities share tokens, count overlaps, and filter.

        Returns
        -------
        dict
            Mapping: source1_entity_id → list of candidate entity_ids.
        """
        candidates = {}

        for _, row in tqdm(df_source1.iterrows(), total=len(df_source1), desc="Address blocking"):
            eid = row[id_col]
            addr = str(row[addr_col])

            query_tokens = tokenize_address(addr)
            query_postals = extract_postal_tokens(addr)

            # Count token overlaps with each target entity
            overlap_counts: Dict[str, int] = {}

            for token in query_tokens:
                if token in self.token_to_entities:
                    for target_eid in self.token_to_entities[token]:
                        overlap_counts[target_eid] = overlap_counts.get(target_eid, 0) + 1

            # Collect candidates meeting the threshold
            cands = set()

            # Postal code match — strong signal
            if self.postal_match_boost:
                for postal in query_postals:
                    if postal in self.postal_to_entities:
                        cands.update(self.postal_to_entities[postal])

            # Token overlap threshold
            for target_eid, count in overlap_counts.items():
                if count >= self.min_token_overlap:
                    cands.add(target_eid)

            # Sort by overlap count (descending) and cap
            cand_list = sorted(
                cands,
                key=lambda x: overlap_counts.get(x, 0),
                reverse=True,
            )[:self.max_candidates_per_query]

            candidates[eid] = cand_list

        return candidates
