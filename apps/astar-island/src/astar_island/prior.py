"""Offline prior model training and calibration.

Two backends: LightGBM (default) and multinomial logistic regression (fallback).
Trained on historical (initial_grid, ground_truth) pairs with entropy weighting.
Includes grouped cross-validation (by round), temperature-scaling calibration,
floor application, and joblib serialization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from astar_island.config import Config
from astar_island.features import extract_features
from astar_island.types import CODE_TO_CLASS, STATIC_CODES, H, K, W
from astar_island.utils import entropy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classifier protocol
# ---------------------------------------------------------------------------


class _Classifier(Protocol):
    """Minimal sklearn-compatible multiclass classifier interface."""

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
        sample_weight: NDArray[np.float64] | None = ...,
    ) -> _Classifier: ...

    def predict_proba(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _softmax_rows(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable row-wise softmax. (N, K) → (N, K)."""
    shifted: NDArray[np.float64] = logits - np.max(logits, axis=1, keepdims=True)
    exp_vals: NDArray[np.float64] = np.exp(shifted)
    return np.asarray(
        exp_vals / np.sum(exp_vals, axis=1, keepdims=True), dtype=np.float64
    )


def _scale_temperature(
    raw_proba: NDArray[np.float64],
    temperature: float,
    eps: float,
) -> NDArray[np.float64]:
    """Apply temperature scaling: softmax(log(p) / tau)."""
    if temperature == 1.0:
        return raw_proba
    log_probs: NDArray[np.float64] = np.log(np.clip(raw_proba, eps, 1.0))
    return _softmax_rows(log_probs / temperature)


def _nll(
    proba: NDArray[np.float64],
    labels: NDArray[np.int64],
    eps: float = 1e-9,
) -> float:
    """Mean negative log-likelihood for hard integer labels.

    Handles the case where predict_proba returns fewer columns than K
    (e.g., when a fold didn't see all classes). Labels referencing
    missing columns get probability `eps` (worst case).
    """
    n = len(labels)
    n_cols = proba.shape[1]
    row_idx = np.arange(n)
    # Clamp labels to valid column range; out-of-range labels get eps
    valid_mask: NDArray[np.bool_] = labels < n_cols
    prob_correct: NDArray[np.float64] = np.full(n, eps, dtype=np.float64)
    if valid_mask.any():
        safe_labels = np.clip(labels, 0, n_cols - 1)
        prob_correct[valid_mask] = np.clip(
            proba[row_idx[valid_mask], safe_labels[valid_mask]],
            eps,
            1.0,
        )
    return float(-np.mean(np.log(prob_correct)))


# ---------------------------------------------------------------------------
# PriorModel
# ---------------------------------------------------------------------------


class PriorModel:
    """Learned prior model for terrain class prediction."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._model: _Classifier | None = None
        self._temperature: float = config.temperature
        self._feature_names: list[str] | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        features: NDArray[np.float64],
        labels: NDArray[np.int32],
        round_ids: NDArray[np.int32],
        sample_weights: NDArray[np.float64] | None = None,
    ) -> None:
        """Train the model using grouped-by-round cross-validation.

        Args:
            features: (N, n_features) feature matrix.
            labels: (N,) integer class labels 0-5.
            round_ids: (N,) round identifier for each sample (for GroupKFold).
            sample_weights: (N,) optional entropy-based weights.
        """
        import warnings

        from sklearn.model_selection import GroupKFold

        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
        )

        y = np.asarray(labels, dtype=np.int64)
        w: NDArray[np.float64] | None = (
            np.asarray(sample_weights, dtype=np.float64)
            if sample_weights is not None
            else None
        )

        unique_rounds = np.unique(round_ids)
        n_splits = min(5, len(unique_rounds))
        gkf = GroupKFold(n_splits=n_splits)

        cv_nll_scores: list[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(
            gkf.split(features, y, groups=round_ids),
        ):
            x_tr = features[train_idx]
            y_tr = y[train_idx]
            w_tr: NDArray[np.float64] | None = w[train_idx] if w is not None else None

            x_val = features[val_idx]
            y_val = y[val_idx]

            fold_model = self._build_model()
            fold_model.fit(x_tr, y_tr, sample_weight=w_tr)

            raw_val: NDArray[np.float64] = np.asarray(
                fold_model.predict_proba(x_val),
                dtype=np.float64,
            )

            # Pad to K columns if fold model didn't see all classes
            n_cols_val = raw_val.shape[1]
            if n_cols_val < K:
                padded_val: NDArray[np.float64] = np.full(
                    (raw_val.shape[0], K),
                    1e-9,
                    dtype=np.float64,
                )
                fold_classes = getattr(fold_model, "classes_", np.arange(n_cols_val))
                for ci, cls_id in enumerate(fold_classes):
                    if 0 <= cls_id < K:
                        padded_val[:, int(cls_id)] = raw_val[:, ci]
                raw_val = padded_val

            # NLL ≈ weighted KL(one_hot(y) || q) for hard labels
            fold_nll = _nll(raw_val, y_val)
            cv_nll_scores.append(fold_nll)
            logger.debug("Fold %d/%d NLL: %.4f", fold_idx + 1, n_splits, fold_nll)

        logger.info(
            "GroupKFold CV mean NLL: %.4f ± %.4f",
            float(np.mean(cv_nll_scores)),
            float(np.std(cv_nll_scores)),
        )

        # --- Train final model on all data ---
        self._model = self._build_model()
        self._model.fit(features, y, sample_weight=w)

        from astar_island.features import feature_names

        self._feature_names = feature_names(self.config)

        # --- Temperature calibration on a held-out 20% of rounds ---
        n_val_rounds = max(1, int(round(0.2 * len(unique_rounds))))
        val_rounds_set: set[int] = set(unique_rounds[-n_val_rounds:].tolist())
        val_mask: NDArray[np.bool_] = np.isin(round_ids, list(val_rounds_set))

        if val_mask.any():
            self._calibrate_temperature(features[val_mask], y[val_mask])
        else:
            logger.warning(
                "No validation rounds for temperature calibration; using τ=%.2f",
                self._temperature,
            )

    def _build_model(self) -> _Classifier:
        """Instantiate a fresh classifier according to config."""
        if self.config.prior_model == "logreg":
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced",
                C=1.0,
            )
            return model  # type: ignore[no-any-return]

        import lightgbm as lgb

        lgb_model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=K,
            max_depth=self.config.lgb_max_depth,
            num_leaves=self.config.lgb_num_leaves,
            min_data_in_leaf=self.config.lgb_min_data_in_leaf,
            n_estimators=self.config.lgb_n_estimators,
            learning_rate=self.config.lgb_learning_rate,
            class_weight="balanced",
            verbose=-1,
            n_jobs=-1,
        )
        return lgb_model

    def _calibrate_temperature(
        self,
        val_features: NDArray[np.float64],
        val_labels: NDArray[np.int64],
    ) -> None:
        """Find optimal temperature τ via scalar minimisation of NLL.

        Sets self._temperature in place.
        Optimal τ typically falls in [0.5, 2.0].
        """
        from scipy.optimize import minimize_scalar

        if self._model is None:
            return

        raw_proba: NDArray[np.float64] = np.asarray(
            self._model.predict_proba(val_features),
            dtype=np.float64,
        )

        # Pad to K columns if the model only learned a subset of classes.
        # LightGBM's predict_proba columns correspond to model.classes_,
        # NOT necessarily [0, 1, ..., K-1].
        n_cols = raw_proba.shape[1]
        if n_cols < K:
            padded: NDArray[np.float64] = np.full(
                (raw_proba.shape[0], K),
                1e-9,
                dtype=np.float64,
            )
            classes = getattr(self._model, "classes_", np.arange(n_cols))
            for i, cls_id in enumerate(classes):
                if 0 <= cls_id < K:
                    padded[:, int(cls_id)] = raw_proba[:, i]
            row_sums = np.sum(padded, axis=1, keepdims=True)
            raw_proba = padded / np.maximum(row_sums, 1e-10)

        def objective(tau: float) -> float:
            calibrated = _scale_temperature(raw_proba, tau, self.config.eps)
            return _nll(calibrated, val_labels)

        result = minimize_scalar(objective, bounds=(0.1, 5.0), method="bounded")
        best_tau: float = float(result.x)
        logger.info(
            "Temperature calibration: τ=%.4f  (NLL τ=1.0: %.4f → τ=best: %.4f)",
            best_tau,
            objective(1.0),
            objective(best_tau),
        )
        self._temperature = best_tau

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return calibrated class probabilities. Shape: (N, K=6).

        Handles the case where the underlying model returns fewer than K
        columns (e.g., LightGBM when some classes were absent from training).
        Missing class columns are filled with eps and renormalized.
        """
        import warnings

        if self._model is None:
            msg = "Model not trained — call train() first"
            raise RuntimeError(msg)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )
            raw: NDArray[np.float64] = np.asarray(
                self._model.predict_proba(features),
                dtype=np.float64,
            )

        # Pad to K columns if the model only learned a subset of classes
        n_cols = raw.shape[1]
        if n_cols < K:
            padded: NDArray[np.float64] = np.full(
                (raw.shape[0], K),
                self.config.eps,
                dtype=np.float64,
            )
            # Map model's learned classes to their correct columns
            # LightGBM stores classes in model.classes_
            classes = getattr(self._model, "classes_", np.arange(n_cols))
            for i, cls_id in enumerate(classes):
                if 0 <= cls_id < K:
                    padded[:, int(cls_id)] = raw[:, i]
            # Renormalize
            row_sums = np.sum(padded, axis=1, keepdims=True)
            raw = padded / np.maximum(row_sums, 1e-10)

        return _scale_temperature(raw, self._temperature, self.config.eps)

    def predict_grid(
        self,
        features: NDArray[np.float64],
        initial_grid: NDArray[np.int32],
    ) -> NDArray[np.float64]:
        """Predict and reshape to (H, W, K) grid tensor with floors applied.

        Args:
            features: (H*W, n_features) feature matrix for this grid.
            initial_grid: (H, W) terrain code grid (for floor rule context).

        Returns:
            (H, W, K) probability tensor with structural floors enforced and
            renormalized to sum to 1 along the class axis.
        """
        calibrated = self.predict_proba(features)  # (H*W, K)
        pred: NDArray[np.float64] = calibrated.reshape(H, W, K)

        # --- Static cells → one-hot ---
        for code in STATIC_CODES:
            code_mask: NDArray[np.bool_] = np.asarray(
                initial_grid == code,
                dtype=np.bool_,
            )
            if code_mask.any():
                pred[code_mask] = 0.0
                pred[code_mask, CODE_TO_CLASS[code]] = 1.0

        # --- Dynamic cells: floor rules ---
        ocean_mask: NDArray[np.bool_] = np.asarray(initial_grid == 10, dtype=np.bool_)
        mountain_static_mask: NDArray[np.bool_] = np.asarray(
            initial_grid == 5,
            dtype=np.bool_,
        )
        static_mask: NDArray[np.bool_] = ocean_mask | mountain_static_mask
        dynamic_mask: NDArray[np.bool_] = ~static_mask

        if dynamic_mask.any():
            from scipy.ndimage import distance_transform_cdt

            # Coastal = adjacent to ocean (Chebyshev distance ≤ 1)
            dist_coast: NDArray[np.float64] = np.asarray(
                distance_transform_cdt(~ocean_mask),
                dtype=np.float64,
            )
            non_coastal_dynamic: NDArray[np.bool_] = (dist_coast > 1.0) & dynamic_mask

            # All plausible classes on dynamic cells → at least floor_standard
            for k in range(K):
                pred[dynamic_mask, k] = np.maximum(
                    pred[dynamic_mask, k],
                    self.config.floor_standard,
                )

            # Mountain (class 5) on ANY dynamic cell → cap to floor_impossible
            # (mountains are STATIC_CODES; dynamic cells can't be mountains)
            # Applied AFTER floor_standard so the cap is not undone.
            pred[dynamic_mask, 5] = np.minimum(
                pred[dynamic_mask, 5],
                self.config.floor_impossible,
            )

            # Port (class 2) on non-coastal dynamic cells → cap to floor_impossible
            pred[non_coastal_dynamic, 2] = np.minimum(
                pred[non_coastal_dynamic, 2],
                self.config.floor_impossible,
            )

        # --- Renormalize each cell to sum to 1 ---
        sums: NDArray[np.float64] = np.sum(pred, axis=-1, keepdims=True)
        sums = np.where(sums < 1e-12, 1.0, sums)
        pred = pred / sums

        return np.asarray(pred, dtype=np.float64)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Serialize model to disk using joblib."""
        import joblib

        payload: dict[str, object] = {
            "model": self._model,
            "temperature": self._temperature,
            "config": self.config,
            "feature_names": self._feature_names,
        }
        joblib.dump(payload, path)
        logger.info("PriorModel saved to %s", path)

    @classmethod
    def load(cls, path: Path, config: Config) -> PriorModel:
        """Load serialized model from disk.

        Args:
            path: Path to the joblib file created by save().
            config: Config instance (floor values and eps come from this).

        Returns:
            A PriorModel with restored weights and calibrated temperature.
        """
        import joblib

        payload = joblib.load(path)
        instance = cls(config)
        instance._model = payload["model"]
        instance._temperature = float(payload["temperature"])
        instance._feature_names = payload.get("feature_names")  # type: ignore[assignment]
        logger.info(
            "PriorModel loaded from %s (τ=%.4f)",
            path,
            instance._temperature,
        )
        return instance

    # ------------------------------------------------------------------
    # Legacy / convenience
    # ------------------------------------------------------------------

    def set_temperature(self, t: float) -> None:
        """Manually override calibration temperature."""
        self._temperature = t

    def predict(self, initial_grid: NDArray[np.int32]) -> NDArray[np.float64]:
        """Convenience: extract features and predict grid in one call.

        Returns: (H, W, K) probability tensor (same as predict_grid).
        """
        feats = extract_features(initial_grid, self.config)
        return self.predict_grid(feats, initial_grid)


# ---------------------------------------------------------------------------
# prepare_training_data
# ---------------------------------------------------------------------------


def prepare_training_data(
    rounds: dict[int, list[tuple[NDArray[np.int32], NDArray[np.float64]]]],
    config: Config,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.float64],
]:
    """Convert loaded round data to training arrays.

    Args:
        rounds: Mapping of round_id → list of (initial_grid, ground_truth) pairs.
                ground_truth shape: (H, W, K) probability distributions.
        config: Config for feature extraction parameters.

    Returns:
        Tuple (features, labels, round_ids, sample_weights) where:
          - features:       (N, n_features) float64
          - labels:         (N,) int32 — argmax of ground_truth
          - round_ids:      (N,) int32 — round identifier for GroupKFold
          - sample_weights: (N,) float64 — entropy of ground_truth distribution

        Only dynamic cells are included (ocean=10 and mountain=5 excluded).
    """
    feats_list: list[NDArray[np.float64]] = []
    labels_list: list[NDArray[np.int32]] = []
    round_ids_list: list[NDArray[np.int32]] = []
    weights_list: list[NDArray[np.float64]] = []

    for round_id, seed_pairs in rounds.items():
        for grid, ground_truth in seed_pairs:
            # Exclude static terrain codes (ocean=10, mountain=5)
            static_mask: NDArray[np.bool_] = np.asarray(
                (grid == 10) | (grid == 5),
                dtype=np.bool_,
            )
            dynamic_flat: NDArray[np.bool_] = ~static_mask.ravel()

            x_all = extract_features(grid, config)  # (H*W, n_features)

            gt_flat: NDArray[np.float64] = ground_truth.reshape(-1, K)
            x_dyn: NDArray[np.float64] = x_all[dynamic_flat]
            gt_dyn: NDArray[np.float64] = gt_flat[dynamic_flat]

            y_hard: NDArray[np.int32] = np.asarray(
                np.argmax(gt_dyn, axis=-1),
                dtype=np.int32,
            )
            cell_weights: NDArray[np.float64] = entropy(gt_dyn)  # (N_dyn,)

            n_dyn = int(dynamic_flat.sum())
            rids: NDArray[np.int32] = np.full(n_dyn, round_id, dtype=np.int32)

            feats_list.append(x_dyn)
            labels_list.append(y_hard)
            round_ids_list.append(rids)
            weights_list.append(cell_weights)

    if not feats_list:
        msg = "No training data provided — rounds dict is empty"
        raise ValueError(msg)

    features_out: NDArray[np.float64] = np.concatenate(feats_list, axis=0)
    labels_out: NDArray[np.int32] = np.concatenate(labels_list, axis=0)
    round_ids_out: NDArray[np.int32] = np.concatenate(round_ids_list, axis=0)
    weights_out: NDArray[np.float64] = np.concatenate(weights_list, axis=0)

    logger.info(
        "prepare_training_data: %d samples from %d unique rounds",
        len(labels_out),
        len(np.unique(round_ids_out)),
    )

    return features_out, labels_out, round_ids_out, weights_out
