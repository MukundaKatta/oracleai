"""Credit scoring simulation.

Demonstrates how AI credit predictions create self-fulfilling default patterns:
1. AI predicts borrower risk
2. Lender uses prediction to set interest rates / deny loans
3. Higher rates increase actual default risk
4. Defaults "validate" the prediction, creating a feedback loop

This is The Oracle's Paradox in action: the prediction of default causes default.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from oracleai.models import SimulationConfig, SimulationResult


@dataclass
class Borrower:
    """A borrower agent in the credit simulation."""

    true_risk: float  # True default probability (unobserved)
    income: float
    credit_history: float  # 0-1, higher = better
    predicted_risk: float = 0.0
    interest_rate: float = 0.05
    loan_approved: bool = True
    defaulted: bool = False


class CreditSimulation:
    """Simulates a lending market with AI-driven credit predictions.

    Demonstrates:
    - How predictions shape outcomes (self-fulfilling prophecy)
    - How feedback loops amplify bias
    - How correction methods can break the loop
    """

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.borrowers: list[Borrower] = []
        self.round_data: list[dict] = []

    def run(self) -> SimulationResult:
        """Run the full credit simulation.

        Returns:
            SimulationResult with performativity analysis.
        """
        self._initialize_borrowers()
        performativity_scores = []

        for round_idx in range(self.config.n_rounds):
            round_info = self._run_round(round_idx)
            performativity_scores.append(round_info["performativity"])
            self.round_data.append(round_info)

        # Compute summary statistics
        true_risks = np.array([b.true_risk for b in self.borrowers])
        predicted_risks = np.array([b.predicted_risk for b in self.borrowers])
        defaults = np.array([b.defaulted for b in self.borrowers], dtype=float)

        # Naive accuracy: how well do predictions match observed defaults
        naive_accuracy = 1.0 - float(np.mean(np.abs(predicted_risks - defaults)))

        # Corrected accuracy: how well do predictions match TRUE risk
        corrected_accuracy = 1.0 - float(np.mean(np.abs(predicted_risks - true_risks)))

        # Bias: difference between predicted and true risk
        bias = float(np.mean(predicted_risks - true_risks))

        return SimulationResult(
            scenario="credit_scoring",
            n_agents=self.config.n_agents,
            n_rounds=self.config.n_rounds,
            performativity_scores=performativity_scores,
            accuracy_naive=naive_accuracy,
            accuracy_corrected=corrected_accuracy,
            bias_introduced=bias,
            details={
                "avg_true_risk": float(np.mean(true_risks)),
                "avg_predicted_risk": float(np.mean(predicted_risks)),
                "avg_default_rate": float(np.mean(defaults)),
                "avg_interest_rate": float(np.mean([b.interest_rate for b in self.borrowers])),
                "loan_denial_rate": float(np.mean([not b.loan_approved for b in self.borrowers])),
            },
        )

    def _initialize_borrowers(self) -> None:
        """Create borrower agents with heterogeneous characteristics."""
        n = self.config.n_agents
        self.borrowers = []

        incomes = self.rng.lognormal(mean=10.5, sigma=0.5, size=n)
        credit_histories = self.rng.beta(a=5, b=2, size=n)

        for i in range(n):
            # True risk is a function of income and credit history
            true_risk = float(np.clip(
                0.3 - 0.15 * credit_histories[i] - 0.1 * (incomes[i] / np.median(incomes))
                + self.rng.normal(0, 0.05),
                0.01, 0.99,
            ))
            self.borrowers.append(Borrower(
                true_risk=true_risk,
                income=float(incomes[i]),
                credit_history=float(credit_histories[i]),
            ))

    def _run_round(self, round_idx: int) -> dict:
        """Run a single round of the simulation."""
        influence = self.config.prediction_influence
        n = len(self.borrowers)

        # Step 1: AI predicts risk
        for b in self.borrowers:
            # Prediction based on observables + noise (initially close to true risk)
            noise = self.rng.normal(0, 0.05)
            # Over time, predictions drift toward observed defaults (feedback loop)
            if round_idx > 0:
                past_default_rate = self.round_data[-1].get("default_rate", b.true_risk)
                b.predicted_risk = float(np.clip(
                    0.5 * (b.true_risk + noise) + 0.5 * past_default_rate,
                    0.01, 0.99,
                ))
            else:
                b.predicted_risk = float(np.clip(b.true_risk + noise, 0.01, 0.99))

        # Step 2: Lender acts on predictions
        for b in self.borrowers:
            # Interest rate increases with predicted risk
            b.interest_rate = 0.03 + 0.2 * b.predicted_risk
            # Loan denied if predicted risk too high
            b.loan_approved = b.predicted_risk < 0.5

        # Step 3: Outcomes are influenced by both true risk AND the prediction
        defaults = []
        for b in self.borrowers:
            if not b.loan_approved:
                # Denied borrowers "default" trivially (can't repay a loan they didn't get)
                # This creates survivorship bias
                b.defaulted = False
                defaults.append(np.nan)
            else:
                # Actual default risk = true risk + influence of higher interest rate
                rate_effect = influence * (b.interest_rate - 0.05) * 2.0
                actual_risk = np.clip(b.true_risk + rate_effect, 0.0, 1.0)
                b.defaulted = self.rng.random() < actual_risk
                defaults.append(float(b.defaulted))

        defaults_arr = np.array([d for d in defaults if not np.isnan(d)])
        predicted_arr = np.array([
            b.predicted_risk for b in self.borrowers if b.loan_approved
        ])

        # Performativity: correlation between prediction and default beyond true risk
        if len(defaults_arr) > 2 and np.std(predicted_arr) > 1e-10:
            perf = float(np.abs(np.corrcoef(predicted_arr, defaults_arr)[0, 1]))
        else:
            perf = 0.0

        return {
            "round": round_idx,
            "default_rate": float(np.nanmean(defaults_arr)) if len(defaults_arr) > 0 else 0.0,
            "performativity": perf,
            "avg_predicted_risk": float(np.mean([b.predicted_risk for b in self.borrowers])),
            "loan_denial_rate": float(np.mean([not b.loan_approved for b in self.borrowers])),
        }
