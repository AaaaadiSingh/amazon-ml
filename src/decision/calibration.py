"""
Calibration + F0.5-optimal threshold search + abstain-to-empty logic.

Key insight from the metric: F0.5 weights precision 2x over recall, so:
1. The optimal threshold is almost certainly ABOVE 0.5
2. Predicting an empty list for a true singleton scores 1.0
3. A false merge on a singleton scores 0.0 — catastrophic

This module implements:
- Isotonic regression calibration of raw model scores
- Grid search for the threshold maximizing macro F0.5
- A separate "confidence floor" for the abstain-to-empty path
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from sklearn.isotonic import IsotonicRegression

from src.eval.metrics import compute_macro_f05


class ScoreCalibrator:
    """
    Isotonic regression calibrator for model scores.

    Maps raw model outputs to well-calibrated probabilities.
    """

    def __init__(self):
        self.calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds='clip'
        )
        self.is_fitted = False

    def fit(self, raw_scores: np.ndarray, labels: np.ndarray):
        """Fit the calibrator on validation predictions and true labels."""
        self.calibrator.fit(raw_scores, labels)
        self.is_fitted = True
        print(f"  Calibrator fitted on {len(raw_scores)} samples")

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        """Transform raw scores to calibrated probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")
        return self.calibrator.transform(raw_scores)


class F05ThresholdOptimizer:
    """
    Grid-search for the decision threshold maximizing macro F0.5.

    Also finds an optional higher "confidence floor" for the abstain path.
    """

    def __init__(
        self,
        threshold_range: Tuple[float, float] = (0.3, 0.95),
        threshold_steps: int = 100,
    ):
        self.threshold_range = threshold_range
        self.threshold_steps = threshold_steps
        self.best_threshold: float = 0.5
        self.best_f05: float = 0.0
        self.search_results: List[Dict] = []

    def search(
        self,
        calibrated_scores: np.ndarray,
        pair_ids: List[Tuple[str, str]],
        ground_truth: pd.DataFrame,
    ) -> float:
        """
        Grid-search for the threshold maximizing macro F0.5.

        Parameters
        ----------
        calibrated_scores : np.ndarray
            Calibrated match probabilities for each candidate pair.
        pair_ids : list of (s1_id, candidate_id)
            Corresponding pair identifiers.
        ground_truth : pd.DataFrame
            Columns: source1_id, matching_ids.

        Returns
        -------
        float
            Best threshold found.
        """
        thresholds = np.linspace(
            self.threshold_range[0],
            self.threshold_range[1],
            self.threshold_steps,
        )

        # Build GT lookup
        gt_lookup = {}
        for _, row in ground_truth.iterrows():
            gt_lookup[row['source1_id']] = set(row['matching_ids'])

        all_s1_ids = list(gt_lookup.keys())

        self.search_results = []
        self.best_f05 = -1
        self.best_threshold = 0.5

        for thresh in thresholds:
            # Build predictions at this threshold
            predictions = self._apply_threshold(calibrated_scores, pair_ids, thresh, all_s1_ids)
            f05 = compute_macro_f05(predictions, gt_lookup)
            self.search_results.append({
                'threshold': thresh,
                'f05': f05,
            })
            if f05 > self.best_f05:
                self.best_f05 = f05
                self.best_threshold = thresh

        print(f"  Threshold search: best={self.best_threshold:.4f}, F0.5={self.best_f05:.4f}")
        print(f"  Threshold range explored: [{self.threshold_range[0]:.2f}, {self.threshold_range[1]:.2f}]")

        return self.best_threshold

    @staticmethod
    def _apply_threshold(
        scores: np.ndarray,
        pair_ids: List[Tuple[str, str]],
        threshold: float,
        all_s1_ids: List[str],
    ) -> Dict[str, List[str]]:
        """Apply a threshold to scores and group by S1 entity."""
        predictions: Dict[str, List[str]] = {s1_id: [] for s1_id in all_s1_ids}

        for i, (s1_id, cand_id) in enumerate(pair_ids):
            if s1_id in predictions and scores[i] >= threshold:
                predictions[s1_id].append(cand_id)

        return predictions


class DecisionLayer:
    """
    Full decision pipeline: calibrate → threshold → abstain.

    Combines calibration and threshold optimization with an explicit
    abstain-to-empty path for low-confidence predictions.
    """

    def __init__(self):
        self.calibrator = ScoreCalibrator()
        self.optimizer = F05ThresholdOptimizer()
        self.match_threshold: float = 0.5
        self.abstain_threshold: Optional[float] = None

    def fit(
        self,
        raw_scores: np.ndarray,
        labels: np.ndarray,
        pair_ids: List[Tuple[str, str]],
        ground_truth: pd.DataFrame,
    ):
        """
        Fit calibration and find optimal threshold.

        Parameters
        ----------
        raw_scores : np.ndarray
            Raw model outputs on validation set.
        labels : np.ndarray
            True binary labels.
        pair_ids : list
            (s1_id, candidate_id) for each score.
        ground_truth : pd.DataFrame
            Full ground truth for computing macro F0.5.
        """
        # Step 1: Calibrate
        self.calibrator.fit(raw_scores, labels)
        calibrated = self.calibrator.transform(raw_scores)

        # Step 2: Find optimal threshold
        self.match_threshold = self.optimizer.search(calibrated, pair_ids, ground_truth)

        # Step 3: Set abstain threshold (slightly above match threshold)
        # The abstain threshold determines: if the BEST candidate for an S1 entity
        # scores below this, predict empty rather than the best candidate.
        # Typically set 5-15% above the match threshold for extra safety.
        self.abstain_threshold = min(self.match_threshold + 0.1, 0.95)

        print(f"  Match threshold: {self.match_threshold:.4f}")
        print(f"  Abstain threshold: {self.abstain_threshold:.4f}")

    def predict(
        self,
        raw_scores: np.ndarray,
        pair_ids: List[Tuple[str, str]],
        all_s1_ids: List[str],
    ) -> Dict[str, List[str]]:
        """
        Make final predictions with calibration + thresholding + abstain.

        Returns
        -------
        dict
            s1_id → list of predicted matching entity IDs (may be empty).
        """
        # Calibrate
        calibrated = self.calibrator.transform(raw_scores)

        # Apply match threshold
        predictions: Dict[str, List[str]] = {s1_id: [] for s1_id in all_s1_ids}
        # Also track best score per S1 entity for abstain logic
        best_score_per_s1: Dict[str, float] = {s1_id: 0.0 for s1_id in all_s1_ids}

        for i, (s1_id, cand_id) in enumerate(pair_ids):
            if s1_id not in predictions:
                predictions[s1_id] = []

            score = calibrated[i]
            if score > best_score_per_s1.get(s1_id, 0.0):
                best_score_per_s1[s1_id] = score

            if score >= self.match_threshold:
                predictions[s1_id].append(cand_id)

        # Abstain-to-empty: if the best candidate score is below
        # the abstain threshold, drop all predictions for that entity
        if self.abstain_threshold is not None:
            n_abstained = 0
            for s1_id in all_s1_ids:
                if (predictions[s1_id] and
                        best_score_per_s1.get(s1_id, 0.0) < self.abstain_threshold):
                    predictions[s1_id] = []
                    n_abstained += 1
            if n_abstained > 0:
                print(f"  Abstained on {n_abstained} entities (best score < {self.abstain_threshold:.4f})")

        n_with = sum(1 for v in predictions.values() if v)
        n_empty = sum(1 for v in predictions.values() if not v)
        print(f"  Predictions: {n_with} with matches, {n_empty} empty/singletons")

        return predictions

    def predict_simple(
        self,
        raw_scores: np.ndarray,
        pair_ids: List[Tuple[str, str]],
        all_s1_ids: List[str],
        threshold: Optional[float] = None,
    ) -> Dict[str, List[str]]:
        """
        Simple threshold-only prediction (no calibration, no abstain).
        Useful for quick baseline evaluation.
        """
        thresh = threshold or self.match_threshold
        predictions: Dict[str, List[str]] = {s1_id: [] for s1_id in all_s1_ids}

        for i, (s1_id, cand_id) in enumerate(pair_ids):
            if s1_id not in predictions:
                predictions[s1_id] = []
            if raw_scores[i] >= thresh:
                predictions[s1_id].append(cand_id)

        return predictions
