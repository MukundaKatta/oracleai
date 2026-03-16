"""CLI entry point for OracleAI.

Provides commands for detection, correction, and simulation of
performative prediction loops.
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from oracleai.models import SimulationConfig

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="oracleai")
def main() -> None:
    """OracleAI — Detect and correct self-fulfilling prophecies in AI prediction systems."""
    pass


@main.command()
@click.option("--data", type=click.Path(exists=True), required=True, help="Path to CSV with predictions data.")
@click.option("--significance", type=float, default=0.05, help="Significance level for detection.")
@click.option("--output", type=click.Path(), default=None, help="Output file for results.")
def detect(data: str, significance: float, output: str | None) -> None:
    """Detect performative prediction loops in data."""
    import pandas as pd
    from oracleai.detection import PerformativeLoopDetector, FeedbackAnalyzer
    from oracleai.detection.metrics import performativity_index

    console.print(Panel("[bold]OracleAI — Performative Loop Detection[/bold]", style="blue"))

    df = pd.read_csv(data)

    required_cols = {"prediction", "outcome", "action"}
    if not required_cols.issubset(df.columns):
        console.print(f"[red]Error: CSV must contain columns: {required_cols}[/red]")
        raise click.Abort()

    predictions = df["prediction"].values
    outcomes = df["outcome"].values
    actions = df["action"].values
    prediction_known = df["prediction_known"].values if "prediction_known" in df.columns else None

    # Run detection
    detector = PerformativeLoopDetector(significance_level=significance)
    result = detector.detect(predictions, outcomes, actions, prediction_known)

    # Run feedback analysis
    analyzer = FeedbackAnalyzer()
    fb_result = analyzer.analyze(predictions, actions, outcomes)

    # Compute performativity index
    pi = performativity_index(predictions, outcomes, prediction_known)

    # Display results
    table = Table(title="Detection Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Performative Loop Detected", str(result.is_performative))
    table.add_row("Loop Type", result.loop_type.value)
    table.add_row("Performativity Score", f"{result.performativity_score:.4f}")
    table.add_row("Performativity Index", f"{pi:.4f}")
    table.add_row("P(outcome | known)", f"{result.p_outcome_known:.4f}")
    table.add_row("P(outcome | unknown)", f"{result.p_outcome_unknown:.4f}")
    table.add_row("Test Statistic", f"{result.statistic:.4f}")
    table.add_row("p-value", f"{result.p_value:.6f}")
    table.add_row("Loop Gain", f"{fb_result.loop_gain:.4f}")
    table.add_row("Loop Stability", fb_result.stability.value)

    console.print(table)

    if output:
        from oracleai.report import generate_detection_report
        generate_detection_report(result, fb_result, pi, Path(output))
        console.print(f"\n[green]Report saved to {output}[/green]")


@main.command()
@click.option("--data", type=click.Path(exists=True), required=True, help="Path to CSV with predictions data.")
@click.option(
    "--method",
    type=click.Choice(["counterfactual", "causal_iv", "causal_did", "causal_psm", "self_aware"]),
    default="counterfactual",
    help="Correction method.",
)
@click.option("--output", type=click.Path(), default=None, help="Output file for corrected predictions.")
def correct(data: str, method: str, output: str | None) -> None:
    """Correct predictions for performative effects."""
    import pandas as pd

    console.print(Panel("[bold]OracleAI — Prediction Correction[/bold]", style="blue"))

    df = pd.read_csv(data)

    predictions = df["prediction"].values
    outcomes = df["outcome"].values

    if method == "counterfactual":
        from oracleai.correction import CounterfactualCorrector

        if "prediction_known" not in df.columns:
            console.print("[red]Error: Counterfactual method requires 'prediction_known' column.[/red]")
            raise click.Abort()

        corrector = CounterfactualCorrector()
        corrector.fit(outcomes, df["prediction_known"].values.astype(bool))
        results = corrector.correct_batch(predictions)

        _display_correction_results(results, "Counterfactual")

    elif method == "self_aware":
        from oracleai.correction import SelfAwarePredictor

        feature_cols = [c for c in df.columns if c not in ("prediction", "outcome", "action", "prediction_known")]
        if not feature_cols:
            console.print("[yellow]Warning: No feature columns found. Using predictions as features.[/yellow]")
            features = predictions.reshape(-1, 1)
        else:
            features = df[feature_cols].values

        predictor = SelfAwarePredictor()
        predictor.fit(features, outcomes, predictions)
        results = predictor.predict_batch(features)

        table = Table(title="Self-Aware Prediction Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Avg Naive Prediction", f"{np.mean([r.naive_prediction for r in results]):.4f}")
        table.add_row("Avg Self-Aware Prediction", f"{np.mean([r.self_aware_prediction for r in results]):.4f}")
        table.add_row("Avg Estimated Influence", f"{np.mean([r.estimated_influence for r in results]):.4f}")
        table.add_row("Convergence Rate", f"{np.mean([r.converged for r in results]):.1%}")
        console.print(table)

    else:
        console.print(f"[yellow]Causal methods require additional data columns. See documentation.[/yellow]")

    if output:
        console.print(f"\n[green]Results saved to {output}[/green]")


@main.command()
@click.option(
    "--scenario",
    type=click.Choice(["credit", "recommendation", "policing"]),
    default="credit",
    help="Simulation scenario.",
)
@click.option("--rounds", type=int, default=100, help="Number of simulation rounds.")
@click.option("--agents", type=int, default=1000, help="Number of agents.")
@click.option("--influence", type=float, default=0.5, help="Prediction influence strength (0-1).")
@click.option("--seed", type=int, default=42, help="Random seed.")
@click.option("--output", type=click.Path(), default=None, help="Output file for results.")
def simulate(
    scenario: str,
    rounds: int,
    agents: int,
    influence: float,
    seed: int,
    output: str | None,
) -> None:
    """Run a performative prediction simulation."""
    from oracleai.simulation import CreditSimulation, RecommendationSimulation, PolicingSimulation

    console.print(Panel(f"[bold]OracleAI — {scenario.title()} Simulation[/bold]", style="blue"))

    config = SimulationConfig(
        n_agents=agents,
        n_rounds=rounds,
        seed=seed,
        prediction_influence=influence,
    )

    sim_map = {
        "credit": CreditSimulation,
        "recommendation": RecommendationSimulation,
        "policing": PolicingSimulation,
    }

    sim = sim_map[scenario](config)

    with console.status(f"Running {scenario} simulation ({rounds} rounds, {agents} agents)..."):
        result = sim.run()

    # Display results
    table = Table(title=f"{scenario.title()} Simulation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Scenario", result.scenario)
    table.add_row("Agents", str(result.n_agents))
    table.add_row("Rounds", str(result.n_rounds))
    table.add_row("Naive Accuracy", f"{result.accuracy_naive:.4f}")
    table.add_row("Corrected Accuracy", f"{result.accuracy_corrected:.4f}")
    table.add_row("Bias Introduced", f"{result.bias_introduced:.4f}")
    table.add_row("Avg Performativity", f"{np.mean(result.performativity_scores):.4f}")
    table.add_row("Final Performativity", f"{result.performativity_scores[-1]:.4f}")

    console.print(table)

    # Show scenario-specific details
    if result.details:
        detail_table = Table(title="Detailed Metrics")
        detail_table.add_column("Metric", style="cyan")
        detail_table.add_column("Value", style="yellow")
        for k, v in result.details.items():
            detail_table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
        console.print(detail_table)

    if output:
        from oracleai.report import generate_simulation_report
        generate_simulation_report(result, Path(output))
        console.print(f"\n[green]Report saved to {output}[/green]")


def _display_correction_results(results: list, method_name: str) -> None:
    """Display correction results in a rich table."""
    table = Table(title=f"{method_name} Correction Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    originals = [r.original for r in results]
    correcteds = [r.corrected for r in results]
    effects = [r.treatment_effect for r in results]

    table.add_row("Avg Original Prediction", f"{np.mean(originals):.4f}")
    table.add_row("Avg Corrected Prediction", f"{np.mean(correcteds):.4f}")
    table.add_row("Avg Treatment Effect", f"{np.mean(effects):.4f}")
    table.add_row("Max Correction", f"{max(abs(o - c) for o, c in zip(originals, correcteds)):.4f}")

    console.print(table)


if __name__ == "__main__":
    main()
