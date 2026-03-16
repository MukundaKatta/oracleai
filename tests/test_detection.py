"""Tests for OracleAI detection module."""

from __future__ import annotations

import numpy as np
import pytest

from oracleai.detection.loop_detector import PerformativeLoopDetector
from oracleai.detection.feedback_analyzer import FeedbackAnalyzer
from oracleai.detection.metrics import (
    counterfactual_gap,
    loop_stability,
    performativity_index,
)
from oracleai.models import FeedbackChain, LoopType, StabilityClass


class TestPerformativeLoopDetector:
    """Tests for the PerformativeLoopDetector."""

    def test_detects_self_fulfilling_loop(self) -> None:
        """When predictions cause outcomes, detection should find performativity."""
        rng = np.random.default_rng(42)
        n = 500

        # Self-fulfilling setup: prediction -> action -> outcome matches prediction
        predictions = rng.random(n)
        actions = predictions + rng.normal(0, 0.1, n)  # Actions follow predictions
        prediction_known = np.array([True] * 250 + [False] * 250)

        # When prediction is known: outcome tracks prediction
        outcomes = np.zeros(n)
        outcomes[:250] = predictions[:250] + rng.normal(0, 0.1, 250)
        # When prediction is unknown: outcome is more random
        outcomes[250:] = rng.random(250)
        outcomes = np.clip(outcomes, 0, 1)

        detector = PerformativeLoopDetector(significance_level=0.05)
        result = detector.detect(predictions, outcomes, actions, prediction_known)

        assert result.is_performative is True
        assert result.performativity_score > 0.2

    def test_no_loop_when_independent(self) -> None:
        """When predictions and outcomes are independent, no loop should be detected."""
        rng = np.random.default_rng(123)
        n = 500

        predictions = rng.random(n)
        outcomes = rng.random(n)  # Completely independent
        actions = rng.random(n)
        prediction_known = rng.choice([True, False], size=n)

        detector = PerformativeLoopDetector(significance_level=0.05)
        result = detector.detect(predictions, outcomes, actions, prediction_known)

        # Should not be highly performative
        assert result.performativity_score < 0.5

    def test_result_fields(self) -> None:
        """Verify all result fields are populated correctly."""
        rng = np.random.default_rng(99)
        n = 100

        predictions = rng.random(n)
        outcomes = rng.random(n)
        actions = rng.random(n)
        known = np.array([True] * 50 + [False] * 50)

        detector = PerformativeLoopDetector()
        result = detector.detect(predictions, outcomes, actions, known)

        assert 0.0 <= result.performativity_score <= 1.0
        assert 0.0 <= result.p_outcome_known <= 1.0
        assert 0.0 <= result.p_outcome_unknown <= 1.0
        assert 0.0 <= result.p_value <= 1.0
        assert result.loop_type in LoopType

    def test_insufficient_samples(self) -> None:
        """With too few samples in one group, should return neutral."""
        predictions = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        outcomes = np.array([1.0, 1.0, 0.0, 1.0, 0.0])
        actions = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        known = np.array([True, True, True, True, True])  # All known, no control

        detector = PerformativeLoopDetector()
        result = detector.detect(predictions, outcomes, actions, known)

        assert result.loop_type == LoopType.NEUTRAL


class TestFeedbackAnalyzer:
    """Tests for the FeedbackAnalyzer."""

    def test_convergent_loop(self) -> None:
        """A damped feedback loop should be classified as convergent."""
        n = 50
        t = np.arange(n)

        # Converging oscillation
        predictions = 0.5 + 0.3 * np.exp(-0.1 * t) * np.cos(0.5 * t)
        outcomes = 0.5 + 0.25 * np.exp(-0.1 * t) * np.cos(0.5 * t + 0.2)
        actions = 0.5 * (predictions + outcomes)

        analyzer = FeedbackAnalyzer()
        result = analyzer.analyze(predictions, actions, outcomes)

        assert result.stability in (StabilityClass.CONVERGENT, StabilityClass.OSCILLATING)
        assert len(result.chains) == n

    def test_loop_gain_computation(self) -> None:
        """Loop gain should be non-negative."""
        rng = np.random.default_rng(42)
        n = 100

        predictions = rng.random(n)
        actions = 0.8 * predictions + 0.2 * rng.random(n)
        outcomes = 0.5 * actions + 0.5 * rng.random(n)

        analyzer = FeedbackAnalyzer()
        result = analyzer.analyze(predictions, actions, outcomes)

        assert result.loop_gain >= 0.0

    def test_chains_built_correctly(self) -> None:
        """Each chain should have correct timestep and values."""
        predictions = np.array([0.1, 0.2, 0.3])
        actions = np.array([0.15, 0.25, 0.35])
        outcomes = np.array([0.12, 0.22, 0.32])

        analyzer = FeedbackAnalyzer()
        result = analyzer.analyze(predictions, actions, outcomes)

        assert len(result.chains) == 3
        assert result.chains[0].timestep == 0
        assert result.chains[2].timestep == 2
        assert abs(result.chains[0].prediction - 0.1) < 1e-10


class TestMetrics:
    """Tests for performativity metrics."""

    def test_performativity_index_range(self) -> None:
        """Performativity index should be in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            preds = rng.random(100)
            outs = rng.random(100)
            pi = performativity_index(preds, outs)
            assert 0.0 <= pi <= 1.0

    def test_performativity_index_high_for_correlated(self) -> None:
        """Highly correlated predictions and outcomes should have high PI."""
        rng = np.random.default_rng(42)
        preds = rng.random(200)
        outcomes = preds + rng.normal(0, 0.05, 200)
        outcomes = np.clip(outcomes, 0, 1)

        pi = performativity_index(preds, outcomes)
        assert pi > 0.3

    def test_loop_stability_convergent(self) -> None:
        """A converging chain should be classified as convergent."""
        chains = [
            FeedbackChain(
                prediction=0.5 + 0.3 * np.exp(-0.1 * t),
                action=0.5,
                outcome=0.5 + 0.1 * np.exp(-0.1 * t),
                validation=0.9,
                timestep=t,
            )
            for t in range(30)
        ]
        result = loop_stability(chains)
        assert result == StabilityClass.CONVERGENT

    def test_counterfactual_gap_zero_when_equal(self) -> None:
        """Gap should be zero when observed equals counterfactual."""
        observed = np.array([1.0, 2.0, 3.0])
        counterfactual = np.array([1.0, 2.0, 3.0])
        assert counterfactual_gap(observed, counterfactual) == pytest.approx(0.0)

    def test_counterfactual_gap_positive(self) -> None:
        """Gap should be positive when arrays differ."""
        observed = np.array([1.0, 2.0, 3.0])
        counterfactual = np.array([0.0, 1.0, 2.0])
        assert counterfactual_gap(observed, counterfactual) == pytest.approx(1.0)

    def test_counterfactual_gap_length_mismatch(self) -> None:
        """Should raise ValueError for mismatched lengths."""
        with pytest.raises(ValueError):
            counterfactual_gap(np.array([1.0, 2.0]), np.array([1.0]))
