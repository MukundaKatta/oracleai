"""Predictive policing simulation.

Demonstrates the feedback loop in predictive policing:
1. AI predicts crime hotspots
2. Police deploy to predicted areas (surveillance increases)
3. More surveillance -> more detected crime (not more actual crime)
4. Detected crime data "validates" prediction
5. Next round: even more prediction -> even more surveillance

The Oracle's Paradox: "Crime will happen here" becomes true because
you looked here (and didn't look elsewhere).
"""

from __future__ import annotations

import numpy as np

from oracleai.models import SimulationConfig, SimulationResult


class PolicingSimulation:
    """Simulates predictive policing feedback loop.

    Models:
    - A grid of neighborhoods with true (latent) crime rates
    - AI predictions of crime per neighborhood
    - Police allocation based on predictions
    - Detection rate proportional to police presence
    - Feedback: detected crime -> updated predictions
    """

    def __init__(
        self,
        config: SimulationConfig | None = None,
        n_neighborhoods: int = 20,
    ) -> None:
        self.config = config or SimulationConfig()
        self.n_neighborhoods = n_neighborhoods
        self.rng = np.random.default_rng(self.config.seed)

        self.true_crime_rates: np.ndarray = np.array([])
        self.predicted_crime: np.ndarray = np.array([])
        self.police_allocation: np.ndarray = np.array([])
        self.detected_crime: np.ndarray = np.array([])
        self.round_data: list[dict] = []

    def run(self) -> SimulationResult:
        """Run the predictive policing simulation."""
        self._initialize()
        performativity_scores = []

        for round_idx in range(self.config.n_rounds):
            round_info = self._run_round(round_idx)
            performativity_scores.append(round_info["performativity"])
            self.round_data.append(round_info)

        # Bias analysis: compare detected crime distribution to true crime distribution
        true_dist = self.true_crime_rates / self.true_crime_rates.sum()
        detected_dist = self.detected_crime / max(self.detected_crime.sum(), 1e-10)
        distribution_bias = float(np.sum(np.abs(detected_dist - true_dist)) / 2.0)

        # Naive accuracy: how well predictions match detected crime
        pred_norm = self.predicted_crime / max(self.predicted_crime.sum(), 1e-10)
        naive_accuracy = 1.0 - float(np.mean(np.abs(pred_norm - detected_dist)))

        # Corrected accuracy: how well predictions match TRUE crime
        corrected_accuracy = 1.0 - float(np.mean(np.abs(pred_norm - true_dist)))

        return SimulationResult(
            scenario="predictive_policing",
            n_agents=self.n_neighborhoods,
            n_rounds=self.config.n_rounds,
            performativity_scores=performativity_scores,
            accuracy_naive=naive_accuracy,
            accuracy_corrected=corrected_accuracy,
            bias_introduced=distribution_bias,
            details={
                "true_crime_gini": float(self._gini(self.true_crime_rates)),
                "detected_crime_gini": float(self._gini(self.detected_crime)),
                "police_allocation_gini": float(self._gini(self.police_allocation)),
                "over_policed_neighborhoods": int(np.sum(
                    self.police_allocation > np.mean(self.police_allocation) * 1.5
                )),
                "under_policed_neighborhoods": int(np.sum(
                    self.police_allocation < np.mean(self.police_allocation) * 0.5
                )),
            },
        )

    def _initialize(self) -> None:
        """Initialize neighborhoods with varying true crime rates."""
        # True crime rates: most neighborhoods are safe, a few have higher crime
        self.true_crime_rates = self.rng.beta(a=2, b=8, size=self.n_neighborhoods)

        # Initial predictions: close to true rates with some noise
        self.predicted_crime = self.true_crime_rates + self.rng.normal(
            0, 0.02, size=self.n_neighborhoods
        )
        self.predicted_crime = np.clip(self.predicted_crime, 0.01, 1.0)

        # Initial uniform police allocation
        total_police = self.n_neighborhoods  # 1 unit per neighborhood on average
        self.police_allocation = np.ones(self.n_neighborhoods) * total_police / self.n_neighborhoods

        self.detected_crime = np.zeros(self.n_neighborhoods)

    def _run_round(self, round_idx: int) -> dict:
        """Run a single round of the policing simulation."""
        influence = self.config.prediction_influence

        # Step 1: Allocate police based on predictions (the performative action)
        total_police = self.n_neighborhoods
        # Police allocation proportional to predicted crime
        pred_weights = self.predicted_crime / max(self.predicted_crime.sum(), 1e-10)
        # Blend between uniform and prediction-based allocation
        uniform = np.ones(self.n_neighborhoods) / self.n_neighborhoods
        self.police_allocation = total_police * (
            (1 - influence) * uniform + influence * pred_weights
        )

        # Step 2: Crime occurs according to TRUE rates (independent of surveillance)
        actual_crime = self.rng.poisson(
            lam=self.true_crime_rates * 100, size=self.n_neighborhoods
        ).astype(float)

        # Step 3: Detection rate depends on police presence (the bias mechanism)
        # Base detection rate + boost from police presence
        base_detection = 0.3
        police_per_capita = self.police_allocation / max(np.mean(self.police_allocation), 1e-10)
        detection_rate = np.clip(
            base_detection + 0.4 * (police_per_capita - 1.0), 0.1, 0.95
        )

        # Detected crime = actual crime * detection rate
        self.detected_crime = actual_crime * detection_rate

        # Step 4: Update predictions based on detected crime (the feedback)
        if round_idx > 0:
            detected_rate = self.detected_crime / 100.0
            # Predictions shift toward detected crime rates
            self.predicted_crime = (
                0.3 * self.predicted_crime + 0.7 * detected_rate
            )
            self.predicted_crime = np.clip(self.predicted_crime, 0.01, 1.0)

        # Performativity: correlation between prediction and detected crime
        # beyond what true crime would explain
        if np.std(self.predicted_crime) > 1e-10 and np.std(self.detected_crime) > 1e-10:
            perf = float(np.abs(
                np.corrcoef(self.predicted_crime, self.detected_crime)[0, 1]
            ))
        else:
            perf = 0.0

        return {
            "round": round_idx,
            "performativity": perf,
            "total_actual_crime": float(actual_crime.sum()),
            "total_detected_crime": float(self.detected_crime.sum()),
            "detection_ratio": float(self.detected_crime.sum() / max(actual_crime.sum(), 1)),
            "allocation_entropy": float(-np.sum(
                pred_weights * np.log(pred_weights + 1e-10)
            )),
        }

    @staticmethod
    def _gini(values: np.ndarray) -> float:
        """Compute Gini coefficient of inequality."""
        values = np.abs(values)
        if values.sum() == 0:
            return 0.0
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        index = np.arange(1, n + 1)
        return float((2 * np.sum(index * sorted_vals) - (n + 1) * np.sum(sorted_vals)) / (n * np.sum(sorted_vals)))
