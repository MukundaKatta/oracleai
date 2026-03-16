"""Self-aware prediction: a predictor that models its own influence on outcomes.

The core idea: instead of predicting P(outcome), predict P(outcome | my prediction = p).
This creates a fixed-point problem: find p* such that p* = P(outcome | prediction = p*).

The self-aware predictor iterates:
1. Make naive prediction p_0
2. Estimate influence: how does knowing p_0 change the outcome?
3. Adjust prediction: p_1 = P(outcome | prediction = p_0)
4. Repeat until convergence to fixed point p*
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from oracleai.models import SelfAwarePrediction


class SelfAwarePredictor:
    """A predictor that accounts for its own influence on outcomes.

    Models the feedback loop explicitly: the prediction changes the world,
    and the predictor knows this. Finds the fixed-point prediction where
    the prediction is self-consistent with its own influence.
    """

    def __init__(
        self,
        base_predictor: Callable[[np.ndarray], float] | None = None,
        influence_function: Callable[[float, np.ndarray], float] | None = None,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        damping: float = 0.5,
    ) -> None:
        """Initialize the self-aware predictor.

        Args:
            base_predictor: Function mapping features -> naive prediction.
                If None, uses a simple linear model.
            influence_function: Function mapping (prediction, features) -> influence on outcome.
                Models how the prediction changes the outcome.
                If None, uses a learned influence model.
            max_iterations: Maximum iterations for fixed-point convergence.
            tolerance: Convergence tolerance.
            damping: Damping factor for iterative updates (0 < damping <= 1).
                Lower = more stable but slower convergence.
        """
        self.base_predictor = base_predictor
        self.influence_function = influence_function
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.damping = damping
        self._influence_slope: float = 0.0
        self._influence_intercept: float = 0.0
        self._is_fitted: bool = False

    def fit(
        self,
        features: np.ndarray,
        outcomes: np.ndarray,
        predictions_history: np.ndarray | None = None,
    ) -> SelfAwarePredictor:
        """Learn the influence function from historical data.

        If prediction history is available, learns how predictions affected outcomes.
        Otherwise, estimates influence from feature-outcome relationships.

        Args:
            features: Feature matrix (shape: [n_samples, n_features]).
            outcomes: Observed outcomes.
            predictions_history: Historical predictions (if available).

        Returns:
            self (for method chaining).
        """
        features = np.asarray(features, dtype=np.float64)
        outcomes = np.asarray(outcomes, dtype=np.float64)

        if features.ndim == 1:
            features = features.reshape(-1, 1)

        # Learn base predictor if not provided
        if self.base_predictor is None:
            # Simple linear model
            X = np.hstack([np.ones((len(features), 1)), features])
            self._base_beta, _, _, _ = np.linalg.lstsq(X, outcomes, rcond=None)
            self.base_predictor = lambda f: float(
                np.hstack([np.ones((1, 1)), np.atleast_2d(f)]) @ self._base_beta
            )

        # Learn influence function if not provided
        if self.influence_function is None and predictions_history is not None:
            predictions_history = np.asarray(predictions_history, dtype=np.float64)
            # Simple model: outcome = alpha + beta * prediction + gamma * features + noise
            # Influence = beta * prediction
            n = len(outcomes)
            X = np.hstack([
                np.ones((n, 1)),
                predictions_history.reshape(-1, 1),
                features,
            ])
            beta, _, _, _ = np.linalg.lstsq(X, outcomes, rcond=None)
            self._influence_slope = float(beta[1])
            self._influence_intercept = 0.0
            self.influence_function = lambda p, f: self._influence_slope * p

        elif self.influence_function is None:
            # Default: assume moderate influence
            self._influence_slope = 0.3
            self.influence_function = lambda p, f: self._influence_slope * p

        self._is_fitted = True
        return self

    def predict(self, features: np.ndarray) -> SelfAwarePrediction:
        """Make a self-aware prediction.

        Iterates to find the fixed-point prediction that is self-consistent
        with its own estimated influence.

        Args:
            features: Feature vector for a single instance.

        Returns:
            SelfAwarePrediction with naive and self-aware predictions.
        """
        if not self._is_fitted and self.base_predictor is None:
            raise RuntimeError("Must call fit() or provide base_predictor and influence_function.")

        features = np.asarray(features, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Step 1: Naive prediction (ignoring own influence)
        assert self.base_predictor is not None
        naive = self.base_predictor(features)

        # Step 2: Iterate to fixed point
        assert self.influence_function is not None
        p = naive
        converged = False

        for i in range(self.max_iterations):
            # Estimate how this prediction influences the outcome
            influence = self.influence_function(p, features)

            # The "true" outcome (without our influence) would be:
            # outcome_true = outcome_observed - influence
            # So our corrected prediction should be:
            p_new = naive - influence + self.influence_function(naive, features)

            # Damped update for stability
            p_new = (1 - self.damping) * p + self.damping * p_new

            if abs(p_new - p) < self.tolerance:
                converged = True
                p = p_new
                break

            p = p_new

        iterations = i + 1
        influence_at_fixed_point = self.influence_function(p, features)

        return SelfAwarePrediction(
            naive_prediction=naive,
            self_aware_prediction=p,
            estimated_influence=influence_at_fixed_point,
            iterations_to_convergence=iterations,
            converged=converged,
        )

    def predict_batch(self, features: np.ndarray) -> list[SelfAwarePrediction]:
        """Make self-aware predictions for a batch of instances.

        Args:
            features: Feature matrix (shape: [n_samples, n_features]).

        Returns:
            List of SelfAwarePrediction objects.
        """
        features = np.asarray(features, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(-1, 1)

        return [self.predict(features[i]) for i in range(len(features))]
