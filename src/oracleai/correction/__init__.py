"""Correction module — methods for correcting performative predictions."""

from oracleai.correction.counterfactual import CounterfactualCorrector
from oracleai.correction.causal import CausalCorrector
from oracleai.correction.self_aware import SelfAwarePredictor

__all__ = [
    "CounterfactualCorrector",
    "CausalCorrector",
    "SelfAwarePredictor",
]
