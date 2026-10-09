"""
Embedding blocker: multilingual bi-encoder + FAISS ANN search.

This is the key generalization lever — ties India/US/France together via
subword semantics rather than hard-coded country logic.

NOTE: The embedding pass is the most compute-intensive step. Plan to run it
on a GPU (Colab free tier or paid SageMaker notebook), cache the embeddings,
and load them on CPU for subsequent runs.
"""

import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import faiss
except ImportError:
    faiss = None
    print("WARNING: faiss not installed. Embedding blocker will not work.")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    print("WARNING: sentence-transformers not installed. Embedding blocker will not work.")


class EmbeddingBlocker:
    """
    Multilingual sentence-embedding blocker with FAISS ANN search.

    Encodes 'name + address' into dense vectors, builds a FAISS index over
    Source 2/3, and retrieves top-K nearest neighbors for each Source 1 entity.
    """

    def __init__(
        self,
        model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        top_k: int = 20,
        batch_size: int = 256,
        use_gpu: bool = False,
    ):
        """
        Parameters
        ----------
        model_name : str
            Sentence-transformer model name. Must be multilingual, MIT/Apache-2.0.
            Good options:
            - 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' (lightweight, Apache-2.0)
            - 'sentence-transformers/LaBSE' (heavier, better cross-lingual, Apache-2.0)
        top_k : int
            Number of nearest neighbors to retrieve per query.
        batch_size : int
            Encoding batch size.
        use_gpu : bool
            Whether to use GPU for encoding.
        """
        self.model_name = model_name
        self.top_k = top_k
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.model = None
        self.index = None
        self.target_ids: List[str] = []
        self.target_embeddings: Optional[np.ndarray] = None

    def _load_model(self):
        """Lazy-load the sentence transformer model."""
        if self.model is None:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers is required for embedding blocking")
            device = 'cuda' if self.use_gpu else 'cpu'
            self.model = SentenceTransformer(self.model_name, device=device)

    def _make_text(self, row: pd.Series, name_col: str, addr_col: str) -> str:
        """Combine name and address into a single string for embedding."""
        name = str(row.get(name_col, ''))
        addr = str(row.get(addr_col, ''))
        return f"{name} {addr}".strip()

    def encode_targets(
        self,
        df_targets: pd.DataFrame,
        name_col: str = 'name_normalized',
        addr_col: str = 'address_normalized',
        id_col: str = 'entity_id',
        cache_path: Optional[str] = None,
    ):
        """
        Encode Source 2/3 records and build the FAISS index.

        Parameters
        ----------
        df_targets : pd.DataFrame
            Combined Source 2 + Source 3 records.
        cache_path : str, optional
            Path to save/load cached embeddings (saves GPU time on re-runs).
        """
        self.target_ids = df_targets[id_col].tolist()

        # Try loading from cache
        if cache_path and os.path.exists(cache_path):
            print(f"  Loading cached target embeddings from {cache_path}")
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            self.target_embeddings = cached['embeddings']
            self.target_ids = cached['ids']
        else:
            self._load_model()
            texts = [self._make_text(row, name_col, addr_col) for _, row in df_targets.iterrows()]
            print(f"  Encoding {len(texts)} target records...")
            self.target_embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
            )

            # Cache if path provided
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'embeddings': self.target_embeddings,
                        'ids': self.target_ids,
                    }, f)
                print(f"  Cached target embeddings to {cache_path}")

        # Build FAISS index
        self._build_faiss_index()

    def _build_faiss_index(self):
        """Build a FAISS index over target embeddings."""
        if faiss is None:
            raise ImportError("faiss is required for embedding blocking")

        dim = self.target_embeddings.shape[1]
        # Use inner product (cosine similarity) since embeddings are normalized
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.target_embeddings.astype(np.float32))
        print(f"  FAISS index built: {self.index.ntotal} vectors, dim={dim}")

    def query(
        self,
        df_source1: pd.DataFrame,
        name_col: str = 'name_normalized',
        addr_col: str = 'address_normalized',
        id_col: str = 'entity_id',
        cache_path: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """
        Find top-K nearest S2/S3 neighbors for each S1 entity.

        Parameters
        ----------
        df_source1 : pd.DataFrame
            Source 1 records.
        cache_path : str, optional
            Path to cache Source 1 embeddings.

        Returns
        -------
        dict
            Mapping: source1_entity_id → list of candidate entity_ids.
        """
        if self.index is None:
            raise RuntimeError("Must call encode_targets() before query()")

        s1_ids = df_source1[id_col].tolist()

        # Encode Source 1
        if cache_path and os.path.exists(cache_path):
            print(f"  Loading cached Source 1 embeddings from {cache_path}")
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            s1_embeddings = cached['embeddings']
        else:
            self._load_model()
            texts = [self._make_text(row, name_col, addr_col) for _, row in df_source1.iterrows()]
            print(f"  Encoding {len(texts)} Source 1 records...")
            s1_embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump({'embeddings': s1_embeddings, 'ids': s1_ids}, f)

        # FAISS search
        print(f"  Searching FAISS index (top-{self.top_k})...")
        distances, indices = self.index.search(
            s1_embeddings.astype(np.float32), self.top_k
        )

        # Build candidate dict
        candidates = {}
        for i, s1_id in enumerate(s1_ids):
            cands = []
            for j in range(self.top_k):
                idx = indices[i, j]
                if idx >= 0:  # -1 means no result
                    cands.append(self.target_ids[idx])
            candidates[s1_id] = cands

        return candidates

    def get_embedding_cosine(
        self,
        s1_id: str,
        s2_id: str,
        s1_embeddings_map: Optional[Dict[str, np.ndarray]] = None,
    ) -> float:
        """Get cosine similarity between two entities (for feature engineering)."""
        if s2_id in self.target_ids:
            idx = self.target_ids.index(s2_id)
            target_vec = self.target_embeddings[idx]
        else:
            return 0.0

        if s1_embeddings_map and s1_id in s1_embeddings_map:
            source_vec = s1_embeddings_map[s1_id]
        else:
            return 0.0

        return float(np.dot(source_vec, target_vec))
