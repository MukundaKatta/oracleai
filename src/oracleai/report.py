"""Report generation for OracleAI analysis results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from oracleai.models import (
    FeedbackLoopResult,
    LoopDetectionResult,
    SimulationResult,
)


def generate_detection_report(
    detection: LoopDetectionResult,
    feedback: FeedbackLoopResult,
    performativity_index: float,
    output_path: Path,
) -> None:
    """Generate a detection report as a text file.

    Args:
        detection: Loop detection result.
        feedback: Feedback analysis result.
        performativity_index: Computed performativity index.
        output_path: Path to write the report.
    """
    lines = [
        "=" * 72,
        "ORACLEAI — Performative Prediction Detection Report",
        "=" * 72,
        "",
        "SUMMARY",
        "-" * 40,
        f"Performative Loop Detected: {detection.is_performative}",
        f"Loop Type:                  {detection.loop_type.value}",
        f"Performativity Score:       {detection.performativity_score:.4f}",
        f"Performativity Index:       {performativity_index:.4f}",
        "",
        "STATISTICAL TEST",
        "-" * 40,
        f"P(outcome | prediction known):   {detection.p_outcome_known:.4f}",
        f"P(outcome | prediction unknown): {detection.p_outcome_unknown:.4f}",
        f"Test Statistic:                  {detection.statistic:.4f}",
        f"p-value:                         {detection.p_value:.6f}",
        f"Confidence:                      {detection.confidence:.4f}",
        "",
        "FEEDBACK LOOP ANALYSIS",
        "-" * 40,
        f"Loop Gain:        {feedback.loop_gain:.4f}",
        f"Stability:        {feedback.stability.value}",
        f"Convergence Rate: {feedback.convergence_rate:.6f}" if feedback.convergence_rate is not None else "Convergence Rate: N/A",
        f"Fixed Point:      {feedback.fixed_point:.4f}" if feedback.fixed_point is not None else "Fixed Point:      N/A",
        "",
        "INTERPRETATION",
        "-" * 40,
    ]

    if detection.is_performative:
        if detection.loop_type.value == "self_fulfilling":
            lines.append(
                "WARNING: Self-fulfilling prophecy detected. Predictions are causing "
                "the outcomes they predict. Consider applying counterfactual or "
                "causal correction methods."
            )
        elif detection.loop_type.value == "self_defeating":
            lines.append(
                "Self-defeating prophecy detected. Predictions are preventing "
                "the outcomes they predict. This may be desirable (e.g., preventive "
                "interventions) but should be monitored."
            )
    else:
        lines.append(
            "No significant performative effect detected. Predictions appear "
            "to be observational rather than causal."
        )

    if feedback.loop_gain > 1.0:
        lines.append(
            f"\nCAUTION: Feedback loop gain ({feedback.loop_gain:.2f}) exceeds 1.0, "
            "indicating an amplifying loop that may destabilize over time."
        )

    lines.extend(["", "=" * 72])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def generate_simulation_report(
    result: SimulationResult,
    output_path: Path,
) -> None:
    """Generate a simulation report.

    Args:
        result: Simulation result.
        output_path: Path to write the report.
    """
    perf_scores = result.performativity_scores

    lines = [
        "=" * 72,
        f"ORACLEAI — {result.scenario.replace('_', ' ').title()} Simulation Report",
        "=" * 72,
        "",
        "CONFIGURATION",
        "-" * 40,
        f"Scenario:    {result.scenario}",
        f"Agents:      {result.n_agents}",
        f"Rounds:      {result.n_rounds}",
        "",
        "RESULTS",
        "-" * 40,
        f"Naive Accuracy:     {result.accuracy_naive:.4f}",
        f"Corrected Accuracy: {result.accuracy_corrected:.4f}",
        f"Accuracy Gap:       {abs(result.accuracy_naive - result.accuracy_corrected):.4f}",
        f"Bias Introduced:    {result.bias_introduced:.4f}",
        "",
        "PERFORMATIVITY OVER TIME",
        "-" * 40,
        f"Initial:  {perf_scores[0]:.4f}" if perf_scores else "Initial:  N/A",
        f"Final:    {perf_scores[-1]:.4f}" if perf_scores else "Final:    N/A",
        f"Mean:     {np.mean(perf_scores):.4f}" if perf_scores else "Mean:     N/A",
        f"Max:      {np.max(perf_scores):.4f}" if perf_scores else "Max:      N/A",
        f"Trend:    {'increasing' if len(perf_scores) > 1 and perf_scores[-1] > perf_scores[0] else 'decreasing or stable'}",
        "",
    ]

    if result.details:
        lines.append("DETAILED METRICS")
        lines.append("-" * 40)
        for k, v in result.details.items():
            if isinstance(v, float):
                lines.append(f"{k:30s} {v:.4f}")
            else:
                lines.append(f"{k:30s} {v}")
        lines.append("")

    lines.append("INTERPRETATION")
    lines.append("-" * 40)

    if result.bias_introduced > 0.1:
        lines.append(
            f"Significant bias ({result.bias_introduced:.2f}) introduced by "
            "performative prediction effects. The prediction system is materially "
            "altering the distribution of outcomes."
        )
    else:
        lines.append(
            "Minimal performative bias detected. The prediction system's influence "
            "on outcomes appears limited."
        )

    gap = abs(result.accuracy_naive - result.accuracy_corrected)
    if gap > 0.05:
        lines.append(
            f"\nThe accuracy gap ({gap:.2f}) between naive and corrected measures "
            "indicates that the system's apparent accuracy is inflated by "
            "self-fulfilling prophecy effects."
        )

    lines.extend(["", "=" * 72])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
