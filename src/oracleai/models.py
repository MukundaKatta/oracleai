"""Data models for OracleAI."""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class LoopType(str, Enum):
    """Type of performative prediction loop."""

    SELF_FULFILLING = "self_fulfilling"
    SELF_DEFEATING = "self_defeating"
    NEUTRAL = "neutral"


class StabilityClass(str, Enum):
    """Stability classification for feedback loops."""

    CONVERGENT = "convergent"
    DIVERGENT = "divergent"
    OSCILLATING = "oscillating"
    UNKNOWN = "unknown"


class LoopDetectionResult(BaseModel):
    """Result from performative loop detection."""

    is_performative: bool = Field(description="Whether a performative loop was detected")
    loop_type: LoopType = Field(description="Type of loop detected")
    performativity_score: float = Field(
        ge=0.0, le=1.0,
        description="Score from 0 (not performative) to 1 (fully self-fulfilling)",
    )
    p_outcome_known: float = Field(
        ge=0.0, le=1.0,
        description="P(outcome | prediction known)",
    )
    p_outcome_unknown: float = Field(
        ge=0.0, le=1.0,
        description="P(outcome | prediction unknown)",
    )
    statistic: float = Field(description="Test statistic")
    p_value: float = Field(description="p-value of the performativity test")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence level (1 - p_value)",
    )
    details: dict[str, Any] = Field(default_factory=dict, description="Additional details")

    model_config = {"arbitrary_types_allowed": True}


class FeedbackChain(BaseModel):
    """A single feedback chain: prediction -> action -> outcome -> validation."""

    prediction: float = Field(description="The prediction value")
    action: float = Field(description="Action taken based on prediction")
    outcome: float = Field(description="Observed outcome")
    validation: float = Field(description="How the outcome validated the prediction")
    timestep: int = Field(default=0, description="Time step in the chain")


class FeedbackLoopResult(BaseModel):
    """Result from feedback loop analysis."""

    chains: list[FeedbackChain] = Field(description="Detected feedback chains")
    loop_gain: float = Field(description="Gain of the feedback loop (>1 = amplifying)")
    stability: StabilityClass = Field(description="Stability classification")
    convergence_rate: float | None = Field(
        default=None,
        description="Rate of convergence (negative = converging, positive = diverging)",
    )
    fixed_point: float | None = Field(
        default=None,
        description="Fixed point of the loop, if convergent",
    )
    details: dict[str, Any] = Field(default_factory=dict)


class CorrectedPrediction(BaseModel):
    """A prediction corrected for performative effects."""

    original: float = Field(description="Original (performative) prediction")
    corrected: float = Field(description="Corrected prediction")
    treatment_effect: float = Field(description="Estimated treatment effect of prediction")
    confidence_interval: tuple[float, float] = Field(
        description="95% confidence interval for corrected prediction",
    )
    method: str = Field(description="Correction method used")


class CausalPrediction(BaseModel):
    """A prediction corrected via causal inference."""

    original: float = Field(description="Original prediction")
    corrected: float = Field(description="Causally corrected prediction")
    causal_effect: float = Field(description="Estimated causal effect")
    confounding_bias: float = Field(description="Estimated confounding bias")
    method: str = Field(description="Causal method used (iv, did, psm)")
    confidence_interval: tuple[float, float] = Field(
        description="95% confidence interval",
    )
    details: dict[str, Any] = Field(default_factory=dict)


class SelfAwarePrediction(BaseModel):
    """A prediction from a self-aware predictor that models its own influence."""

    naive_prediction: float = Field(description="Prediction ignoring own influence")
    self_aware_prediction: float = Field(description="Prediction accounting for own influence")
    estimated_influence: float = Field(description="Estimated influence on outcome")
    iterations_to_convergence: int = Field(description="Iterations to reach fixed point")
    converged: bool = Field(description="Whether the iterative process converged")


class SimulationConfig(BaseModel):
    """Configuration for running a simulation."""

    n_agents: int = Field(default=1000, ge=10, description="Number of agents")
    n_rounds: int = Field(default=100, ge=1, description="Number of simulation rounds")
    seed: int = Field(default=42, description="Random seed")
    prediction_influence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="How strongly predictions influence outcomes",
    )


class SimulationResult(BaseModel):
    """Result from a simulation run."""

    scenario: str = Field(description="Simulation scenario name")
    n_agents: int = Field(description="Number of agents")
    n_rounds: int = Field(description="Number of rounds run")
    performativity_scores: list[float] = Field(description="Performativity score per round")
    accuracy_naive: float = Field(description="Naive accuracy (treating predictions as ground truth)")
    accuracy_corrected: float = Field(description="Accuracy after correction")
    bias_introduced: float = Field(description="Bias introduced by performative effects")
    details: dict[str, Any] = Field(default_factory=dict)
