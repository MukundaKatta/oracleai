"""Causal prediction correction.

Uses causal inference methods to separate genuine predictive signal from
self-fulfilling prophecy effects. Implements three approaches:
- Instrumental Variables (IV) estimation
- Difference-in-Differences (DiD)
- Propensity Score Matching (PSM)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from oracleai.models import CausalPrediction


class CausalCorrector:
    """Separates genuine prediction from self-fulfillment using causal inference.

    The key insight: in a performative system, the observed correlation between
    predictions and outcomes conflates two effects:
    1. The prediction captures real signal about the outcome (genuine prediction)
    2. The prediction causes the outcome through behavior change (performative effect)

    This class isolates (1) by removing (2).
    """

    def __init__(
        self,
        method: str = "iv",
        confidence_level: float = 0.95,
    ) -> None:
        """Initialize the causal corrector.

        Args:
            method: Causal inference method.
                'iv' = Instrumental Variables,
                'did' = Difference-in-Differences,
                'psm' = Propensity Score Matching.
            confidence_level: Confidence level for intervals.
        """
        if method not in ("iv", "did", "psm"):
            raise ValueError(f"Unknown method: {method}. Use 'iv', 'did', or 'psm'.")
        self.method = method
        self.confidence_level = confidence_level

    def correct(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        confounders: np.ndarray | None = None,
        instruments: np.ndarray | None = None,
        pre_treatment_outcomes: np.ndarray | None = None,
        treatment: np.ndarray | None = None,
    ) -> CausalPrediction:
        """Correct predictions using causal inference.

        Args:
            predictions: Array of predictions.
            outcomes: Array of observed outcomes.
            confounders: Confounder matrix (required for PSM).
            instruments: Instrumental variables (required for IV).
            pre_treatment_outcomes: Pre-treatment outcomes (required for DiD).
            treatment: Binary treatment indicator (required for DiD and PSM).

        Returns:
            CausalPrediction with corrected values.
        """
        predictions = np.asarray(predictions, dtype=np.float64)
        outcomes = np.asarray(outcomes, dtype=np.float64)

        if self.method == "iv":
            return self._iv_estimation(predictions, outcomes, instruments)
        elif self.method == "did":
            return self._did_estimation(
                predictions, outcomes, pre_treatment_outcomes, treatment
            )
        elif self.method == "psm":
            return self._psm_estimation(predictions, outcomes, confounders, treatment)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _iv_estimation(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        instruments: np.ndarray | None,
    ) -> CausalPrediction:
        """Two-Stage Least Squares (2SLS) IV estimation.

        Stage 1: Regress predictions on instruments (Z -> P_hat)
        Stage 2: Regress outcomes on predicted predictions (P_hat -> Y)

        The instrument must:
        1. Be correlated with the prediction (relevance)
        2. Not directly affect the outcome (exclusion restriction)
        """
        if instruments is None:
            raise ValueError("IV method requires instruments.")

        instruments = np.asarray(instruments, dtype=np.float64)
        if instruments.ndim == 1:
            instruments = instruments.reshape(-1, 1)

        n = len(predictions)
        Z = np.hstack([np.ones((n, 1)), instruments])

        # Stage 1: P = Z @ gamma + e
        gamma, _, _, _ = np.linalg.lstsq(Z, predictions, rcond=None)
        p_hat = Z @ gamma

        # Stage 2: Y = alpha + beta * P_hat + u
        X2 = np.column_stack([np.ones(n), p_hat])
        beta, _, _, _ = np.linalg.lstsq(X2, outcomes, rcond=None)
        causal_effect = float(beta[1])

        # Naive OLS for comparison
        X_naive = np.column_stack([np.ones(n), predictions])
        beta_naive, _, _, _ = np.linalg.lstsq(X_naive, outcomes, rcond=None)
        naive_effect = float(beta_naive[1])

        confounding_bias = naive_effect - causal_effect

        # Standard errors via asymptotic approximation
        residuals = outcomes - X2 @ beta
        mse = float(np.sum(residuals**2) / max(n - 2, 1))
        try:
            cov = mse * np.linalg.inv(X2.T @ X2)
            se = float(np.sqrt(cov[1, 1]))
        except np.linalg.LinAlgError:
            se = float(np.std(residuals) / np.sqrt(n))

        z_val = stats.norm.ppf((1 + self.confidence_level) / 2)
        corrected_mean = float(np.mean(predictions)) * causal_effect
        ci = (corrected_mean - z_val * se, corrected_mean + z_val * se)

        return CausalPrediction(
            original=float(np.mean(predictions)),
            corrected=corrected_mean,
            causal_effect=causal_effect,
            confounding_bias=confounding_bias,
            method="iv",
            confidence_interval=ci,
            details={
                "naive_effect": naive_effect,
                "first_stage_f": self._first_stage_f(instruments, predictions),
                "n_instruments": instruments.shape[1],
            },
        )

    def _did_estimation(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        pre_treatment_outcomes: np.ndarray | None,
        treatment: np.ndarray | None,
    ) -> CausalPrediction:
        """Difference-in-Differences estimation.

        Compares changes over time between a treatment group (predictions known)
        and a control group (predictions unknown).

        DiD estimator = (Y_treat_post - Y_treat_pre) - (Y_control_post - Y_control_pre)
        """
        if pre_treatment_outcomes is None:
            raise ValueError("DiD method requires pre_treatment_outcomes.")
        if treatment is None:
            raise ValueError("DiD method requires treatment indicator.")

        pre = np.asarray(pre_treatment_outcomes, dtype=np.float64)
        treatment = np.asarray(treatment, dtype=bool)

        treat_mask = treatment
        ctrl_mask = ~treatment

        if treat_mask.sum() < 2 or ctrl_mask.sum() < 2:
            raise ValueError("Need at least 2 observations in each group.")

        # DiD estimator
        treat_diff = np.mean(outcomes[treat_mask]) - np.mean(pre[treat_mask])
        ctrl_diff = np.mean(outcomes[ctrl_mask]) - np.mean(pre[ctrl_mask])
        did_estimate = treat_diff - ctrl_diff

        # Naive effect
        naive_effect = float(np.mean(outcomes[treat_mask]) - np.mean(outcomes[ctrl_mask]))
        confounding_bias = naive_effect - did_estimate

        # Standard error via clustered bootstrap approximation
        n_treat = treat_mask.sum()
        n_ctrl = ctrl_mask.sum()
        se_treat = np.std(outcomes[treat_mask] - pre[treat_mask], ddof=1) / np.sqrt(n_treat)
        se_ctrl = np.std(outcomes[ctrl_mask] - pre[ctrl_mask], ddof=1) / np.sqrt(n_ctrl)
        se = float(np.sqrt(se_treat**2 + se_ctrl**2))

        z_val = stats.norm.ppf((1 + self.confidence_level) / 2)
        corrected_mean = float(np.mean(predictions)) - did_estimate
        ci = (corrected_mean - z_val * se, corrected_mean + z_val * se)

        return CausalPrediction(
            original=float(np.mean(predictions)),
            corrected=corrected_mean,
            causal_effect=did_estimate,
            confounding_bias=confounding_bias,
            method="did",
            confidence_interval=ci,
            details={
                "treat_diff": float(treat_diff),
                "ctrl_diff": float(ctrl_diff),
                "parallel_trends_test": float(
                    stats.ttest_ind(pre[treat_mask], pre[ctrl_mask]).pvalue
                ),
            },
        )

    def _psm_estimation(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        confounders: np.ndarray | None,
        treatment: np.ndarray | None,
    ) -> CausalPrediction:
        """Propensity Score Matching estimation.

        1. Estimate propensity scores: P(treatment | confounders)
        2. Match treated to control units with similar propensity scores
        3. Estimate treatment effect from matched sample
        """
        if confounders is None:
            raise ValueError("PSM method requires confounders.")
        if treatment is None:
            raise ValueError("PSM method requires treatment indicator.")

        confounders = np.asarray(confounders, dtype=np.float64)
        treatment = np.asarray(treatment, dtype=bool)

        if confounders.ndim == 1:
            confounders = confounders.reshape(-1, 1)

        # Step 1: Estimate propensity scores
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(confounders, treatment.astype(int))
        propensity = lr.predict_proba(confounders)[:, 1]

        # Step 2: Match using nearest neighbors on propensity scores
        treated_idx = np.where(treatment)[0]
        control_idx = np.where(~treatment)[0]

        if len(treated_idx) < 1 or len(control_idx) < 1:
            raise ValueError("Need observations in both treatment and control groups.")

        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(propensity[control_idx].reshape(-1, 1))
        distances, indices = nn.kneighbors(propensity[treated_idx].reshape(-1, 1))
        matched_control_idx = control_idx[indices.flatten()]

        # Step 3: ATT from matched sample
        treated_outcomes = outcomes[treated_idx]
        matched_outcomes = outcomes[matched_control_idx]
        att = float(np.mean(treated_outcomes - matched_outcomes))

        # Naive effect
        naive_effect = float(np.mean(outcomes[treatment]) - np.mean(outcomes[~treatment]))
        confounding_bias = naive_effect - att

        # Standard error
        se = float(np.std(treated_outcomes - matched_outcomes, ddof=1) / np.sqrt(len(treated_idx)))
        z_val = stats.norm.ppf((1 + self.confidence_level) / 2)
        corrected_mean = float(np.mean(predictions)) - att
        ci = (corrected_mean - z_val * se, corrected_mean + z_val * se)

        return CausalPrediction(
            original=float(np.mean(predictions)),
            corrected=corrected_mean,
            causal_effect=att,
            confounding_bias=confounding_bias,
            method="psm",
            confidence_interval=ci,
            details={
                "n_matched": len(treated_idx),
                "avg_match_distance": float(np.mean(distances)),
                "propensity_overlap": float(
                    min(propensity[treatment].max(), propensity[~treatment].max())
                    - max(propensity[treatment].min(), propensity[~treatment].min())
                ),
            },
        )

    def _first_stage_f(
        self, instruments: np.ndarray, predictions: np.ndarray
    ) -> float:
        """Compute the first-stage F-statistic for IV relevance."""
        n = len(predictions)
        Z = np.hstack([np.ones((n, 1)), instruments])
        gamma, _, _, _ = np.linalg.lstsq(Z, predictions, rcond=None)
        p_hat = Z @ gamma
        residuals = predictions - p_hat

        ssr = np.sum((p_hat - np.mean(predictions)) ** 2)
        sse = np.sum(residuals**2)
        k = instruments.shape[1]

        if sse < 1e-12:
            return float("inf")

        f_stat = (ssr / k) / (sse / max(n - k - 1, 1))
        return float(f_stat)
