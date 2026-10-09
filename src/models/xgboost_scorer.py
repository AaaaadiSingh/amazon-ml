"""
XGBoost scorer for entity matching (Stage A).

Design choices:
- Uses xgb.train() on DMatrix (not the sklearn wrapper) for direct control.
- Named feature columns (not the SageMaker "no headers, target-first" convention).
- scale_pos_weight handles class imbalance.
- Outputs calibrated probabilities (via isotonic regression in decision module).
"""

import os
import json
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold


class XGBoostScorer:
    """
    XGBoost binary classifier for entity matching.

    Trained on pairwise features from candidate pairs. Outputs
    raw probabilities that are later calibrated in the decision module.
    """

    def __init__(
        self,
        params: Optional[Dict] = None,
        num_rounds: int = 500,
        early_stopping_rounds: int = 50,
    ):
        """
        Parameters
        ----------
        params : dict, optional
            XGBoost parameters. Defaults to a precision-biased configuration.
        num_rounds : int
            Maximum boosting rounds.
        early_stopping_rounds : int
            Stop if validation metric doesn't improve for this many rounds.
        """
        self.params = params or self._default_params()
        self.num_rounds = num_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.model: Optional[xgb.Booster] = None
        self.feature_names: Optional[List[str]] = None

    @staticmethod
    def _default_params() -> Dict:
        """
        Default XGBoost params, biased toward precision.

        - Higher min_child_weight and lower max_depth → more conservative splits
        - eval_metric = 'logloss' for probability calibration quality
        """
        return {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'min_child_weight': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 1.0,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'seed': 42,
            'verbosity': 1,
        }

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        scale_pos_weight: Optional[float] = None,
    ) -> Dict:
        """
        Train the XGBoost model.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix.
        y_train : np.ndarray
            Binary labels (1 = match, 0 = non-match).
        X_val, y_val : optional
            Validation set for early stopping.
        scale_pos_weight : float, optional
            If None, auto-computed from class ratio.

        Returns
        -------
        dict
            Training results including best iteration and feature importances.
        """
        self.feature_names = list(X_train.columns)

        # Auto-compute class weight if not provided
        if scale_pos_weight is None:
            n_pos = y_train.sum()
            n_neg = len(y_train) - n_pos
            if n_pos > 0:
                scale_pos_weight = n_neg / n_pos
            else:
                scale_pos_weight = 1.0

        params = dict(self.params)
        params['scale_pos_weight'] = scale_pos_weight

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)

        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
            evals.append((dval, 'val'))

        print(f"  Training XGBoost: {len(X_train)} samples, "
              f"{len(self.feature_names)} features, "
              f"scale_pos_weight={scale_pos_weight:.2f}")

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.num_rounds,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds if X_val is not None else None,
            verbose_eval=50,
        )

        # Feature importances
        importance = self.model.get_score(importance_type='gain')
        importance_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        results = {
            'best_iteration': self.model.best_iteration if hasattr(self.model, 'best_iteration') else self.num_rounds,
            'feature_importance': importance_sorted,
        }

        print(f"  Best iteration: {results['best_iteration']}")
        print(f"  Top 10 features by gain:")
        for fname, gain in importance_sorted[:10]:
            print(f"    {fname}: {gain:.2f}")

        return results

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict match probabilities.

        Returns
        -------
        np.ndarray
            Predicted probabilities (0-1) for each candidate pair.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        dmat = xgb.DMatrix(X, feature_names=self.feature_names)
        return self.model.predict(dmat)

    def save(self, path: str):
        """Save model and metadata."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        self.model.save_model(path + '.xgb')
        with open(path + '.meta.json', 'w') as f:
            json.dump({
                'feature_names': self.feature_names,
                'params': self.params,
                'num_rounds': self.num_rounds,
            }, f, indent=2)
        print(f"  Model saved to {path}.xgb")

    def load(self, path: str):
        """Load model and metadata."""
        self.model = xgb.Booster()
        self.model.load_model(path + '.xgb')
        with open(path + '.meta.json', 'r') as f:
            meta = json.load(f)
        self.feature_names = meta['feature_names']
        self.params = meta['params']
        print(f"  Model loaded from {path}.xgb")


def cross_validate_scorer(
    X: pd.DataFrame,
    y: np.ndarray,
    pair_ids: List[Tuple[str, str]],
    n_folds: int = 5,
    params: Optional[Dict] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Cross-validate the XGBoost scorer, returning out-of-fold predictions.

    Stratified by label, but NOTE: pairs from the same S1 entity should ideally
    be in the same fold (entity-level splitting). This is a simplified version.

    Returns
    -------
    (oof_predictions, fold_results)
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n  === Fold {fold_idx + 1}/{n_folds} ===")
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        scorer = XGBoostScorer(params=params)
        result = scorer.train(X_tr, y_tr, X_va, y_va)
        preds = scorer.predict(X_va)
        oof_preds[val_idx] = preds
        fold_results.append(result)

    return oof_preds, fold_results
