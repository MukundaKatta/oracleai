"""Counterfactual prediction correction.

Adjusts predictions to account for performative effects using the
potential outcomes framework. Answers: "What would the outcome be
if the prediction had NOT been made (or acted upon)?"
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from oracleai.models import CorrectedPrediction


class CounterfactualCorrector:
    """Adjusts predictions by estimating and removing performative treatment effects.

    Uses the potential outcomes framework:
    - Y(1) = outcome when prediction is known (factual)
    - Y(0) = outcome when prediction is unknown (counterfactual)
    - Treatment effect = E[Y(1)] - E[Y(0)]

    The corrected prediction estimates Y(0): the outcome absent the prediction's influence.
    """

    def __init__(
        self,
        method: str = "ate",
        confidence_level: float = 0.95,
    ) -> None:
        """Initialize the corrector.

        Args:
            method: Treatment effect estimation method.
                'ate' = Average Treatment Effect,
                'att' = Average Treatment Effect on the Treated,
                'regression' = Regression adjustment.
            confidence_level: Confidence level for intervals.
        """
        if method not in ("ate", "att", "regression"):
            raise ValueError(f"Unknown method: {method}. Use 'ate', 'att', or 'regression'.")
        self.method = method
        self.confidence_level = confidence_level
        self._treatment_effect: float | None = None
        self._se: float | None = None

    def fit(
        self,
        outcomes: np.ndarray,
        prediction_known: np.ndarray,
        covariates: np.ndarray | None = None,
    ) -> CounterfactualCorrector:
        """Estimate the treatment effect from historical data.

        Args:
            outcomes: Observed outcomes.
            prediction_known: Boolean array — True if prediction was known.
            covariates: Optional covariate matrix for regression adjustment.

        Returns:
            self (for method chaining).
        """
        outcomes = np.asarray(outcomes, dtype=np.float64)
        prediction_known = np.asarray(prediction_known, dtype=bool)

        treated = outcomes[prediction_known]
        control = outcomes[~prediction_known]

        if len(treated) < 2 or len(control) < 2:
            raise ValueError("Need at least 2 samples in each group.")

        if self.method == "ate":
            self._treatment_effect = float(np.mean(treated) - np.mean(control))
            se_t = np.std(treated, ddof=1) / np.sqrt(len(treated))
            se_c = np.std(control, ddof=1) / np.sqrt(len(control))
            self._se = float(np.sqrt(se_t**2 + se_c**2))

        elif self.method == "att":
            # ATT: effect on those whose predictions were known
            self._treatment_effect = float(np.mean(treated) - np.mean(control))
            self._se = float(np.std(treated, ddof=1) / np.sqrt(len(treated)))

        elif self.method == "regression":
            if covariates is None:
                raise ValueError("Regression method requires covariates.")
            covariates = np.asarray(covariates, dtype=np.float64)
            # OLS with treatment indicator and covariates
            n = len(outcomes)
            treatment = prediction_known.astype(np.float64).reshape(-1, 1)
            if covariates.ndim == 1:
                covariates = covariates.reshape(-1, 1)
            X = np.hstack([np.ones((n, 1)), treatment, covariates])
            try:
                beta = np.linalg.lstsq(X, outcomes, rcond=None)[0]
                self._treatment_effect = float(beta[1])
                residuals = outcomes - X @ beta
                mse = float(np.sum(residuals**2) / max(n - X.shape[1], 1))
                try:
                    cov_matrix = mse * np.linalg.inv(X.T @ X)
                    self._se = float(np.sqrt(cov_matrix[1, 1]))
                except np.linalg.LinAlgError:
                    self._se = float(np.std(residuals, ddof=1) / np.sqrt(n))
            except np.linalg.LinAlgError:
                # Fallback to simple difference
                self._treatment_effect = float(np.mean(treated) - np.mean(control))
                self._se = float(
                    np.sqrt(
                        np.var(treated, ddof=1) / len(treated)
                        + np.var(control, ddof=1) / len(control)
                    )
                )

        return self

    def correct(
        self,
        prediction: float,
        treatment_effect: float | None = None,
    ) -> CorrectedPrediction:
        """Correct a single prediction for performative effects.

        Args:
            prediction: The original (potentially performative) prediction.
            treatment_effect: Override the estimated treatment effect. If None,
                uses the effect estimated in fit().

        Returns:
            CorrectedPrediction with original and corrected values.
        """
        if treatment_effect is None:
            if self._treatment_effect is None:
                raise RuntimeError("Must call fit() before correct(), or provide treatment_effect.")
            treatment_effect = self._treatment_effect

        se = self._se if self._se is not None else abs(treatment_effect) * 0.1

        corrected = prediction - treatment_effect
        z = stats.norm.ppf((1 + self.confidence_level) / 2)
        ci = (corrected - z * se, corrected + z * se)

        return CorrectedPrediction(
            original=prediction,
            corrected=corrected,
            treatment_effect=treatment_effect,
            confidence_interval=ci,
            method=f"counterfactual_{self.method}",
        )

    def correct_batch(
        self,
        predictions: np.ndarray,
        treatment_effect: float | None = None,
    ) -> list[CorrectedPrediction]:
        """Correct a batch of predictions.

        Args:
            predictions: Array of predictions to correct.
            treatment_effect: Optional override for treatment effect.

        Returns:
            List of CorrectedPrediction objects.
        """
        predictions = np.asarray(predictions, dtype=np.float64)
        return [self.correct(float(p), treatment_effect) for p in predictions]
