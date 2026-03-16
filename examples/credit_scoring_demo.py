#!/usr/bin/env python3
"""Credit Scoring Demo — The Oracle's Paradox in Action.

Demonstrates how AI credit predictions create self-fulfilling default patterns
and how OracleAI correction methods can break the loop.

Usage:
    python examples/credit_scoring_demo.py
"""

from __future__ import annotations

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from oracleai.models import SimulationConfig
from oracleai.simulation.credit import CreditSimulation
from oracleai.detection.loop_detector import PerformativeLoopDetector
from oracleai.detection.metrics import performativity_index
from oracleai.correction.counterfactual import CounterfactualCorrector

console = Console()


def main() -> None:
    """Run the credit scoring demo."""
    console.print(Panel(
        "[bold]The Oracle's Paradox: Credit Scoring Demo[/bold]\n\n"
        '"Would you still have defaulted if I hadn\'t predicted it?"',
        style="blue",
    ))

    # --- Step 1: Run simulation ---
    console.print("\n[bold cyan]Step 1: Simulating a lending market with AI predictions[/bold cyan]\n")

    config = SimulationConfig(
        n_agents=500,
        n_rounds=50,
        seed=42,
        prediction_influence=0.6,
    )
    sim = CreditSimulation(config)
    result = sim.run()

    table = Table(title="Simulation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Agents", str(result.n_agents))
    table.add_row("Rounds", str(result.n_rounds))
    table.add_row("Naive Accuracy", f"{result.accuracy_naive:.4f}")
    table.add_row("True Accuracy", f"{result.accuracy_corrected:.4f}")
    table.add_row("Bias Introduced", f"{result.bias_introduced:.4f}")
    table.add_row("Final Performativity", f"{result.performativity_scores[-1]:.4f}")
    console.print(table)

    # --- Step 2: Detect performative loop ---
    console.print("\n[bold cyan]Step 2: Detecting the self-fulfilling prophecy loop[/bold cyan]\n")

    # Extract data from simulation
    rng = np.random.default_rng(42)
    n = len(sim.borrowers)
    predictions = np.array([b.predicted_risk for b in sim.borrowers])
    outcomes = np.array([float(b.defaulted) for b in sim.borrowers])
    actions = np.array([b.interest_rate for b in sim.borrowers])

    # Some borrowers had their predictions known (influenced lender), others not
    prediction_known = np.array([True] * (n // 2) + [False] * (n - n // 2))
    rng.shuffle(prediction_known)

    detector = PerformativeLoopDetector()
    detection = detector.detect(predictions, outcomes, actions, prediction_known)

    pi = performativity_index(predictions, outcomes, prediction_known)

    det_table = Table(title="Detection Results")
    det_table.add_column("Metric", style="cyan")
    det_table.add_column("Value", style="green")
    det_table.add_row("Performative Loop", str(detection.is_performative))
    det_table.add_row("Loop Type", detection.loop_type.value)
    det_table.add_row("Performativity Index", f"{pi:.4f}")
    det_table.add_row("P(default | known)", f"{detection.p_outcome_known:.4f}")
    det_table.add_row("P(default | unknown)", f"{detection.p_outcome_unknown:.4f}")
    det_table.add_row("p-value", f"{detection.p_value:.6f}")
    console.print(det_table)

    # --- Step 3: Correct predictions ---
    console.print("\n[bold cyan]Step 3: Correcting predictions using counterfactual method[/bold cyan]\n")

    corrector = CounterfactualCorrector(method="ate")
    corrector.fit(outcomes, prediction_known)

    # Correct a few example predictions
    examples = [0.3, 0.5, 0.7, 0.9]
    corr_table = Table(title="Corrected Predictions")
    corr_table.add_column("Original", style="red")
    corr_table.add_column("Corrected", style="green")
    corr_table.add_column("Treatment Effect", style="yellow")
    corr_table.add_column("95% CI", style="dim")

    for pred in examples:
        result = corrector.correct(pred)
        corr_table.add_row(
            f"{result.original:.3f}",
            f"{result.corrected:.3f}",
            f"{result.treatment_effect:.3f}",
            f"({result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f})",
        )

    console.print(corr_table)

    # --- Summary ---
    console.print(Panel(
        "[bold]Key Insight:[/bold] The AI's prediction of default risk "
        "CAUSED higher default rates through higher interest rates. "
        "The counterfactual corrector estimates what default rates "
        "would have been WITHOUT the prediction's influence.\n\n"
        "[dim]This is the Oracle's Paradox: the prediction changes the future "
        "it claims to merely forecast.[/dim]",
        style="yellow",
    ))


if __name__ == "__main__":
    main()
