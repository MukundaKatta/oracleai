"""OracleAI — Detecting and correcting self-fulfilling prophecies in AI prediction systems."""

__version__ = "0.1.0"

from oracleai.detection.loop_detector import PerformativeLoopDetector
from oracleai.detection.feedback_analyzer import FeedbackAnalyzer
from oracleai.detection.metrics import performativity_index, loop_stability, counterfactual_gap
from oracleai.correction.counterfactual import CounterfactualCorrector
from oracleai.correction.causal import CausalCorrector
from oracleai.correction.self_aware import SelfAwarePredictor

__all__ = [
    "PerformativeLoopDetector",
    "FeedbackAnalyzer",
    "CounterfactualCorrector",
    "CausalCorrector",
    "SelfAwarePredictor",
    "performativity_index",
    "loop_stability",
    "counterfactual_gap",
]
