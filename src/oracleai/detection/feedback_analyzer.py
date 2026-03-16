"""Analyze feedback cycles in performative prediction systems.

Traces the full cycle: prediction -> action -> outcome -> validation -> updated prediction.
Identifies stable vs unstable feedback loops and computes convergence properties.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from oracleai.models import FeedbackChain, FeedbackLoopResult, StabilityClass


class FeedbackAnalyzer:
    """Traces and analyzes prediction-action-outcome feedback chains.

    A feedback loop exists when:
    1. A prediction P is made
    2. An action A is taken based on P
    3. An outcome O results from A
    4. O is used to validate/update P
    5. The updated P influences future A
    """

    def __init__(self, convergence_threshold: float = 1e-4) -> None:
        """Initialize the analyzer.

        Args:
            convergence_threshold: Threshold for determining convergence.
        """
        self.convergence_threshold = convergence_threshold

    def analyze(
        self,
        predictions: np.ndarray,
        actions: np.ndarray,
        outcomes: np.ndarray,
    ) -> FeedbackLoopResult:
        """Analyze a feedback loop from time-series data.

        Args:
            predictions: Array of predictions over time (shape: [T,]).
            actions: Array of actions taken over time (shape: [T,]).
            outcomes: Array of outcomes observed over time (shape: [T,]).

        Returns:
            FeedbackLoopResult with loop analysis.
        """
        predictions = np.asarray(predictions, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        outcomes = np.asarray(outcomes, dtype=np.float64)

        n = len(predictions)
        if not (len(actions) == n and len(outcomes) == n):
            raise ValueError("All arrays must have the same length.")

        # Build feedback chains
        chains = self._build_chains(predictions, actions, outcomes)

        # Compute loop gain
        loop_gain = self._compute_loop_gain(predictions, actions, outcomes)

        # Assess stability
        stability, convergence_rate = self._assess_stability(predictions, outcomes)

        # Find fixed point if convergent
        fixed_point = self._find_fixed_point(predictions, outcomes, stability)

        return FeedbackLoopResult(
            chains=chains,
            loop_gain=loop_gain,
            stability=stability,
            convergence_rate=convergence_rate,
            fixed_point=fixed_point,
            details={
                "n_timesteps": n,
                "prediction_variance": float(np.var(predictions)),
                "outcome_variance": float(np.var(outcomes)),
                "prediction_outcome_correlation": float(
                    np.corrcoef(predictions, outcomes)[0, 1]
                ) if np.std(predictions) > 1e-10 and np.std(outcomes) > 1e-10 else 0.0,
            },
        )

    def _build_chains(
        self,
        predictions: np.ndarray,
        actions: np.ndarray,
        outcomes: np.ndarray,
    ) -> list[FeedbackChain]:
        """Build feedback chains from time-series data."""
        chains = []
        for t in range(len(predictions)):
            # Validation: how well does the outcome match the prediction?
            validation = 1.0 - abs(predictions[t] - outcomes[t])
            chains.append(
                FeedbackChain(
                    prediction=float(predictions[t]),
                    action=float(actions[t]),
                    outcome=float(outcomes[t]),
                    validation=float(validation),
                    timestep=t,
                )
            )
        return chains

    def _compute_loop_gain(
        self,
        predictions: np.ndarray,
        actions: np.ndarray,
        outcomes: np.ndarray,
    ) -> float:
        """Compute the gain of the feedback loop.

        Loop gain = (dA/dP) * (dO/dA) * (dP'/dO)

        Where:
            dA/dP = how strongly actions respond to predictions
            dO/dA = how strongly outcomes respond to actions
            dP'/dO = how strongly next predictions respond to outcomes

        Gain > 1 means the loop amplifies; gain < 1 means it dampens.
        """
        n = len(predictions)

        # Estimate dA/dP via linear regression
        if np.std(predictions) > 1e-10:
            slope_ap = stats.linregress(predictions, actions).slope
        else:
            slope_ap = 0.0

        # Estimate dO/dA
        if np.std(actions) > 1e-10:
            slope_oa = stats.linregress(actions, outcomes).slope
        else:
            slope_oa = 0.0

        # Estimate dP'/dO (next prediction from current outcome)
        if n > 1 and np.std(outcomes[:-1]) > 1e-10:
            slope_po = stats.linregress(outcomes[:-1], predictions[1:]).slope
        else:
            slope_po = 0.0

        gain = abs(slope_ap * slope_oa * slope_po)
        return float(gain)

    def _assess_stability(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
    ) -> tuple[StabilityClass, float | None]:
        """Assess the stability of the feedback loop.

        Returns:
            Tuple of (stability class, convergence rate).
        """
        n = len(predictions)
        if n < 4:
            return StabilityClass.UNKNOWN, None

        # Compute the prediction-outcome gap over time
        gaps = np.abs(predictions - outcomes)

        # Fit an exponential trend to the gaps
        t = np.arange(n, dtype=np.float64)
        log_gaps = np.log(gaps + 1e-10)

        try:
            result = stats.linregress(t, log_gaps)
            rate = float(result.slope)
        except Exception:
            return StabilityClass.UNKNOWN, None

        # Check for oscillation
        if n >= 6:
            diffs = np.diff(gaps)
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
            oscillation_ratio = sign_changes / max(len(diffs) - 1, 1)
            if oscillation_ratio > 0.6:
                return StabilityClass.OSCILLATING, rate

        if rate < -self.convergence_threshold:
            return StabilityClass.CONVERGENT, rate
        elif rate > self.convergence_threshold:
            return StabilityClass.DIVERGENT, rate
        else:
            return StabilityClass.CONVERGENT, rate

    def _find_fixed_point(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        stability: StabilityClass,
    ) -> float | None:
        """Find the fixed point of a convergent loop.

        The fixed point is where prediction = outcome (the loop is self-consistent).
        """
        if stability not in (StabilityClass.CONVERGENT,):
            return None

        # Use the last few values as estimate of fixed point
        window = min(10, len(predictions) // 4)
        if window < 1:
            return None

        # Average of the last predictions and outcomes
        avg_pred = np.mean(predictions[-window:])
        avg_out = np.mean(outcomes[-window:])
        fixed_point = (avg_pred + avg_out) / 2.0
        return float(fixed_point)
