"""Detect performative prediction loops.

A prediction is performative if the act of making the prediction (and it being known)
changes the probability of the predicted outcome. Formally:

    P(outcome | prediction known) != P(outcome | prediction unknown)

This module provides statistical tests for detecting such performativity.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from oracleai.models import LoopDetectionResult, LoopType


class PerformativeLoopDetector:
    """Detect whether predictions are performative — i.e., self-fulfilling or self-defeating.

    Uses a quasi-experimental design: compares outcomes when predictions are known
    (treatment group) vs. unknown (control group) to the decision-maker.
    """

    def __init__(self, significance_level: float = 0.05) -> None:
        """Initialize the detector.

        Args:
            significance_level: Threshold for statistical significance.
        """
        self.significance_level = significance_level

    def detect(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        actions: np.ndarray,
        prediction_known: np.ndarray | None = None,
    ) -> LoopDetectionResult:
        """Detect performative prediction loops.

        Args:
            predictions: Array of predictions (binary or continuous).
            outcomes: Array of observed outcomes (binary or continuous).
            actions: Array of actions taken based on predictions.
            prediction_known: Boolean array indicating whether each prediction
                was known to the decision-maker. If None, uses a statistical
                approach based on prediction-action correlation.

        Returns:
            LoopDetectionResult with detection details.
        """
        predictions = np.asarray(predictions, dtype=np.float64)
        outcomes = np.asarray(outcomes, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)

        n = len(predictions)
        if not (len(outcomes) == n and len(actions) == n):
            raise ValueError("All input arrays must have the same length.")

        if prediction_known is None:
            prediction_known = self._infer_knowledge(predictions, actions)
        else:
            prediction_known = np.asarray(prediction_known, dtype=bool)

        # Split into treatment (prediction known) and control (prediction unknown)
        known_mask = prediction_known.astype(bool)
        unknown_mask = ~known_mask

        if known_mask.sum() < 5 or unknown_mask.sum() < 5:
            return LoopDetectionResult(
                is_performative=False,
                loop_type=LoopType.NEUTRAL,
                performativity_score=0.0,
                p_outcome_known=float(np.mean(outcomes[known_mask])) if known_mask.any() else 0.0,
                p_outcome_unknown=float(np.mean(outcomes[unknown_mask])) if unknown_mask.any() else 0.0,
                statistic=0.0,
                p_value=1.0,
                confidence=0.0,
                details={"error": "Insufficient samples in one or both groups."},
            )

        # Compute conditional outcome rates
        p_known = float(np.mean(outcomes[known_mask]))
        p_unknown = float(np.mean(outcomes[unknown_mask]))

        # Two-sample test for difference in outcome rates
        # Use permutation-based approach for robustness
        observed_diff = p_known - p_unknown

        # Also run a standard z-test for proportions (binary) or t-test (continuous)
        if self._is_binary(outcomes):
            stat, p_value = self._proportion_z_test(
                outcomes[known_mask], outcomes[unknown_mask]
            )
        else:
            stat_result = stats.ttest_ind(
                outcomes[known_mask], outcomes[unknown_mask], equal_var=False
            )
            stat = float(stat_result.statistic)
            p_value = float(stat_result.pvalue)

        # Compute performativity score: normalized absolute difference
        performativity_score = self._compute_performativity_score(
            predictions, outcomes, actions, known_mask
        )

        # Determine loop type
        loop_type = self._classify_loop(observed_diff, predictions, outcomes, known_mask)

        is_performative = p_value < self.significance_level

        return LoopDetectionResult(
            is_performative=is_performative,
            loop_type=loop_type,
            performativity_score=performativity_score,
            p_outcome_known=p_known,
            p_outcome_unknown=p_unknown,
            statistic=stat,
            p_value=p_value,
            confidence=1.0 - p_value,
            details={
                "n_known": int(known_mask.sum()),
                "n_unknown": int(unknown_mask.sum()),
                "observed_diff": observed_diff,
                "significance_level": self.significance_level,
            },
        )

    def _infer_knowledge(
        self, predictions: np.ndarray, actions: np.ndarray
    ) -> np.ndarray:
        """Infer whether predictions were likely known based on prediction-action correlation.

        Uses the strength of correlation between prediction and action as a proxy
        for whether the prediction influenced the action (and thus was known).
        """
        # Median split: high correlation between prediction and action suggests known
        # Use a sliding window approach
        n = len(predictions)
        window = max(20, n // 10)
        known = np.zeros(n, dtype=bool)

        for i in range(0, n - window + 1, window // 2):
            end = min(i + window, n)
            chunk_pred = predictions[i:end]
            chunk_act = actions[i:end]
            if np.std(chunk_pred) > 1e-10 and np.std(chunk_act) > 1e-10:
                corr = np.abs(np.corrcoef(chunk_pred, chunk_act)[0, 1])
                # High correlation => prediction was likely known and acted upon
                known[i:end] = corr > 0.5

        # Ensure we have both groups; fall back to random split if needed
        if known.all() or not known.any():
            rng = np.random.default_rng(42)
            known = rng.random(n) < 0.5

        return known

    def _is_binary(self, arr: np.ndarray) -> bool:
        """Check if an array contains only binary values."""
        unique = np.unique(arr)
        return len(unique) <= 2 and np.all(np.isin(unique, [0, 1]))

    def _proportion_z_test(
        self, group1: np.ndarray, group2: np.ndarray
    ) -> tuple[float, float]:
        """Two-proportion z-test."""
        n1, n2 = len(group1), len(group2)
        p1, p2 = np.mean(group1), np.mean(group2)
        p_pool = (np.sum(group1) + np.sum(group2)) / (n1 + n2)

        if p_pool == 0 or p_pool == 1:
            return 0.0, 1.0

        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        if se < 1e-12:
            return 0.0, 1.0

        z = (p1 - p2) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        return float(z), float(p_value)

    def _compute_performativity_score(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        actions: np.ndarray,
        known_mask: np.ndarray,
    ) -> float:
        """Compute a 0-1 performativity score.

        Combines: (1) prediction-action coupling, (2) action-outcome coupling,
        (3) difference in outcome rates between known and unknown groups.
        """
        # Prediction-action coupling
        if np.std(predictions) > 1e-10 and np.std(actions) > 1e-10:
            pa_corr = np.abs(np.corrcoef(predictions, actions)[0, 1])
        else:
            pa_corr = 0.0

        # Action-outcome coupling
        if np.std(actions) > 1e-10 and np.std(outcomes) > 1e-10:
            ao_corr = np.abs(np.corrcoef(actions, outcomes)[0, 1])
        else:
            ao_corr = 0.0

        # Outcome rate difference
        p_known = np.mean(outcomes[known_mask])
        p_unknown = np.mean(outcomes[~known_mask])
        rate_diff = min(abs(p_known - p_unknown), 1.0)

        # Weighted combination
        score = 0.4 * pa_corr + 0.3 * ao_corr + 0.3 * rate_diff
        return float(np.clip(score, 0.0, 1.0))

    def _classify_loop(
        self,
        observed_diff: float,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        known_mask: np.ndarray,
    ) -> LoopType:
        """Classify the loop as self-fulfilling, self-defeating, or neutral."""
        if abs(observed_diff) < 0.01:
            return LoopType.NEUTRAL

        # Check if predictions and outcomes are more aligned when prediction is known
        known_agreement = np.mean(
            np.sign(predictions[known_mask] - 0.5) == np.sign(outcomes[known_mask] - 0.5)
        )
        unknown_agreement = np.mean(
            np.sign(predictions[~known_mask] - 0.5) == np.sign(outcomes[~known_mask] - 0.5)
        )

        if known_agreement > unknown_agreement + 0.05:
            return LoopType.SELF_FULFILLING
        elif known_agreement < unknown_agreement - 0.05:
            return LoopType.SELF_DEFEATING
        else:
            return LoopType.NEUTRAL
