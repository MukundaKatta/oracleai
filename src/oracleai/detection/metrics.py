"""Performativity metrics for measuring self-fulfilling prophecy effects.

These metrics quantify the degree to which predictions influence their own outcomes.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from oracleai.models import FeedbackChain, StabilityClass


def performativity_index(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    prediction_known: np.ndarray | None = None,
) -> float:
    """Compute the performativity index: a 0-1 score of how self-fulfilling predictions are.

    The index combines:
    - Prediction-outcome alignment (higher when predictions match outcomes)
    - Temporal coupling (higher when prediction changes precede outcome changes)
    - Differential impact (higher when known predictions have different outcome rates)

    Args:
        predictions: Array of predictions.
        outcomes: Array of outcomes.
        prediction_known: Optional boolean array for known/unknown split.

    Returns:
        Float in [0, 1]. Higher = more performative.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)

    n = len(predictions)
    if n < 2:
        return 0.0

    # 1. Prediction-outcome correlation
    if np.std(predictions) > 1e-10 and np.std(outcomes) > 1e-10:
        corr = np.abs(np.corrcoef(predictions, outcomes)[0, 1])
    else:
        corr = 0.0

    # 2. Temporal Granger-like measure: do prediction changes lead outcome changes?
    if n >= 4:
        pred_diff = np.diff(predictions)
        out_diff = np.diff(outcomes)
        if np.std(pred_diff) > 1e-10 and np.std(out_diff[1:]) > 1e-10 and len(pred_diff) > 1:
            # Lagged correlation: prediction change at t -> outcome change at t+1
            lagged_corr = np.abs(np.corrcoef(pred_diff[:-1], out_diff[1:])[0, 1])
        else:
            lagged_corr = 0.0
    else:
        lagged_corr = 0.0

    # 3. Differential impact (if known/unknown split available)
    if prediction_known is not None:
        prediction_known = np.asarray(prediction_known, dtype=bool)
        if prediction_known.any() and (~prediction_known).any():
            p_known = np.mean(outcomes[prediction_known])
            p_unknown = np.mean(outcomes[~prediction_known])
            diff_impact = min(abs(p_known - p_unknown), 1.0)
        else:
            diff_impact = 0.0
    else:
        diff_impact = corr * 0.5  # Conservative estimate

    # Weighted combination
    index = 0.4 * corr + 0.3 * lagged_corr + 0.3 * diff_impact
    return float(np.clip(index, 0.0, 1.0))


def loop_stability(feedback_chain: list[FeedbackChain]) -> StabilityClass:
    """Assess whether a feedback loop converges or diverges.

    Analyzes the trajectory of prediction-outcome gaps across time steps.

    Args:
        feedback_chain: List of FeedbackChain objects ordered by time.

    Returns:
        StabilityClass enum value.
    """
    if len(feedback_chain) < 4:
        return StabilityClass.UNKNOWN

    gaps = np.array([abs(c.prediction - c.outcome) for c in feedback_chain])
    t = np.arange(len(gaps), dtype=np.float64)

    # Fit exponential trend
    log_gaps = np.log(gaps + 1e-10)
    try:
        result = stats.linregress(t, log_gaps)
        rate = result.slope
    except Exception:
        return StabilityClass.UNKNOWN

    # Check oscillation
    if len(gaps) >= 6:
        diffs = np.diff(gaps)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        oscillation_ratio = sign_changes / max(len(diffs) - 1, 1)
        if oscillation_ratio > 0.6:
            return StabilityClass.OSCILLATING

    if rate < -1e-4:
        return StabilityClass.CONVERGENT
    elif rate > 1e-4:
        return StabilityClass.DIVERGENT
    else:
        return StabilityClass.CONVERGENT


def counterfactual_gap(
    observed: np.ndarray,
    counterfactual: np.ndarray,
) -> float:
    """Compute the counterfactual gap: difference between what happened and what would have.

    This measures the causal impact of the prediction system on outcomes.

    Args:
        observed: Array of observed outcomes (with prediction system).
        counterfactual: Array of counterfactual outcomes (without prediction system).

    Returns:
        Float representing the average absolute difference.
    """
    observed = np.asarray(observed, dtype=np.float64)
    counterfactual = np.asarray(counterfactual, dtype=np.float64)

    if len(observed) != len(counterfactual):
        raise ValueError("Arrays must have the same length.")

    gap = np.mean(np.abs(observed - counterfactual))
    return float(gap)
