"""Detection module — identify performative prediction loops."""

from oracleai.detection.loop_detector import PerformativeLoopDetector
from oracleai.detection.feedback_analyzer import FeedbackAnalyzer
from oracleai.detection.metrics import performativity_index, loop_stability, counterfactual_gap

__all__ = [
    "PerformativeLoopDetector",
    "FeedbackAnalyzer",
    "performativity_index",
    "loop_stability",
    "counterfactual_gap",
]
