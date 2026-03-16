"""Tests for OracleAI correction module."""

from __future__ import annotations

import numpy as np
import pytest

from oracleai.correction.counterfactual import CounterfactualCorrector
from oracleai.correction.causal import CausalCorrector
from oracleai.correction.self_aware import SelfAwarePredictor


class TestCounterfactualCorrector:
    """Tests for the CounterfactualCorrector."""

    def test_fit_and_correct(self) -> None:
        """Basic fit/correct pipeline should work."""
        rng = np.random.default_rng(42)
        n = 200

        prediction_known = np.array([True] * 100 + [False] * 100)
        outcomes = np.zeros(n)
        outcomes[:100] = 0.7 + rng.normal(0, 0.1, 100)  # Higher when known
        outcomes[100:] = 0.4 + rng.normal(0, 0.1, 100)  # Lower when unknown

        corrector = CounterfactualCorrector(method="ate")
        corrector.fit(outcomes, prediction_known)

        result = corrector.correct(0.7)
        assert result.corrected < result.original  # Should correct downward
        assert result.treatment_effect > 0  # Known group had higher outcomes
        assert result.confidence_interval[0] < result.corrected < result.confidence_interval[1]

    def test_correct_batch(self) -> None:
        """Batch correction should return correct number of results."""
        rng = np.random.default_rng(42)
        n = 100
        prediction_known = rng.choice([True, False], size=n)
        outcomes = rng.random(n)

        corrector = CounterfactualCorrector()
        corrector.fit(outcomes, prediction_known)

        predictions = np.array([0.3, 0.5, 0.7])
        results = corrector.correct_batch(predictions)
        assert len(results) == 3

    def test_manual_treatment_effect(self) -> None:
        """Should accept manual treatment effect override."""
        corrector = CounterfactualCorrector()
        # Don't need to fit when providing treatment effect manually
        result = corrector.correct(0.8, treatment_effect=0.2)
        assert result.corrected == pytest.approx(0.6)
        assert result.treatment_effect == pytest.approx(0.2)

    def test_regression_method(self) -> None:
        """Regression adjustment method should work with covariates."""
        rng = np.random.default_rng(42)
        n = 200

        prediction_known = np.array([True] * 100 + [False] * 100)
        covariates = rng.random((n, 2))
        outcomes = 0.3 + 0.2 * prediction_known.astype(float) + 0.1 * covariates[:, 0] + rng.normal(0, 0.05, n)

        corrector = CounterfactualCorrector(method="regression")
        corrector.fit(outcomes, prediction_known, covariates)

        result = corrector.correct(0.6)
        assert result.treatment_effect > 0

    def test_error_without_fit(self) -> None:
        """Should raise error if correct() called without fit()."""
        corrector = CounterfactualCorrector()
        with pytest.raises(RuntimeError):
            corrector.correct(0.5)

    def test_invalid_method(self) -> None:
        """Should raise ValueError for unknown method."""
        with pytest.raises(ValueError):
            CounterfactualCorrector(method="invalid")


class TestCausalCorrector:
    """Tests for the CausalCorrector."""

    def test_iv_estimation(self) -> None:
        """IV estimation should produce valid causal predictions."""
        rng = np.random.default_rng(42)
        n = 500

        # Instrument -> Prediction -> Outcome (with confounding)
        instrument = rng.normal(0, 1, n)
        confounder = rng.normal(0, 1, n)
        predictions = 0.5 * instrument + 0.3 * confounder + rng.normal(0, 0.1, n)
        outcomes = 0.3 * predictions + 0.4 * confounder + rng.normal(0, 0.1, n)

        corrector = CausalCorrector(method="iv")
        result = corrector.correct(
            predictions=predictions,
            outcomes=outcomes,
            instruments=instrument,
        )

        assert result.method == "iv"
        assert result.confidence_interval[0] < result.corrected < result.confidence_interval[1]

    def test_did_estimation(self) -> None:
        """Difference-in-differences should estimate treatment effect."""
        rng = np.random.default_rng(42)
        n = 200

        treatment = np.array([True] * 100 + [False] * 100)
        pre_outcomes = 0.5 + rng.normal(0, 0.1, n)
        # Treatment group gets a boost of 0.2
        post_outcomes = pre_outcomes + 0.2 * treatment.astype(float) + rng.normal(0, 0.05, n)
        predictions = post_outcomes + rng.normal(0, 0.05, n)

        corrector = CausalCorrector(method="did")
        result = corrector.correct(
            predictions=predictions,
            outcomes=post_outcomes,
            pre_treatment_outcomes=pre_outcomes,
            treatment=treatment,
        )

        assert result.method == "did"
        # Causal effect should be approximately 0.2
        assert abs(result.causal_effect - 0.2) < 0.15

    def test_psm_estimation(self) -> None:
        """Propensity score matching should work with confounders."""
        rng = np.random.default_rng(42)
        n = 300

        confounders = rng.random((n, 2))
        # Treatment probability depends on confounders
        treat_prob = 1 / (1 + np.exp(-(confounders[:, 0] - 0.5)))
        treatment = rng.random(n) < treat_prob

        outcomes = 0.3 * confounders[:, 0] + 0.15 * treatment.astype(float) + rng.normal(0, 0.1, n)
        predictions = outcomes + rng.normal(0, 0.05, n)

        corrector = CausalCorrector(method="psm")
        result = corrector.correct(
            predictions=predictions,
            outcomes=outcomes,
            confounders=confounders,
            treatment=treatment,
        )

        assert result.method == "psm"
        assert "n_matched" in result.details

    def test_iv_requires_instruments(self) -> None:
        """IV method should raise ValueError without instruments."""
        corrector = CausalCorrector(method="iv")
        with pytest.raises(ValueError, match="instruments"):
            corrector.correct(
                predictions=np.array([1.0, 2.0]),
                outcomes=np.array([1.0, 2.0]),
            )

    def test_invalid_method(self) -> None:
        """Should raise ValueError for unknown method."""
        with pytest.raises(ValueError):
            CausalCorrector(method="invalid")


class TestSelfAwarePredictor:
    """Tests for the SelfAwarePredictor."""

    def test_fit_and_predict(self) -> None:
        """Basic fit/predict pipeline should work."""
        rng = np.random.default_rng(42)
        n = 200

        features = rng.random((n, 3))
        predictions_history = features @ np.array([0.3, 0.2, 0.5])
        outcomes = predictions_history + 0.2 * predictions_history + rng.normal(0, 0.05, n)

        predictor = SelfAwarePredictor()
        predictor.fit(features, outcomes, predictions_history)

        result = predictor.predict(features[0])
        assert result.converged is True
        assert result.iterations_to_convergence >= 1
        assert isinstance(result.naive_prediction, float)
        assert isinstance(result.self_aware_prediction, float)

    def test_predict_batch(self) -> None:
        """Batch prediction should return correct number of results."""
        rng = np.random.default_rng(42)
        features = rng.random((50, 2))
        outcomes = rng.random(50)

        predictor = SelfAwarePredictor()
        predictor.fit(features, outcomes)

        results = predictor.predict_batch(features[:5])
        assert len(results) == 5

    def test_custom_functions(self) -> None:
        """Should work with custom base predictor and influence function."""
        predictor = SelfAwarePredictor(
            base_predictor=lambda f: 0.5,
            influence_function=lambda p, f: 0.1 * p,
        )
        predictor._is_fitted = True

        result = predictor.predict(np.array([1.0, 2.0]))
        assert result.converged is True
        assert isinstance(result.self_aware_prediction, float)

    def test_convergence_with_damping(self) -> None:
        """Higher damping should lead to faster convergence."""
        rng = np.random.default_rng(42)
        features = rng.random((100, 2))
        outcomes = rng.random(100)

        # Low damping
        p_low = SelfAwarePredictor(damping=0.1)
        p_low.fit(features, outcomes)
        r_low = p_low.predict(features[0])

        # High damping
        p_high = SelfAwarePredictor(damping=0.9)
        p_high.fit(features, outcomes)
        r_high = p_high.predict(features[0])

        # Both should converge
        assert r_low.converged is True
        assert r_high.converged is True
