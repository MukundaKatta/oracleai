"""Content recommendation simulation.

Demonstrates how recommendation algorithms shape preferences over time,
creating filter bubbles and self-fulfilling engagement predictions:
1. Algorithm predicts user preferences
2. Recommends content matching predicted preferences
3. User engages with recommended content (exposure effect)
4. Engagement data "confirms" predictions
5. Preferences genuinely shift toward recommendations

The Oracle's Paradox: "You will like this" becomes true because you were shown it.
"""

from __future__ import annotations

import numpy as np

from oracleai.models import SimulationConfig, SimulationResult


class RecommendationSimulation:
    """Simulates content recommendation with preference drift.

    Models:
    - Users with latent topic preferences (vectors in topic space)
    - Content items with topic distributions
    - Recommendation algorithm that predicts engagement
    - Preference drift: exposure shifts preferences toward consumed content
    """

    def __init__(self, config: SimulationConfig | None = None, n_topics: int = 10) -> None:
        self.config = config or SimulationConfig()
        self.n_topics = n_topics
        self.rng = np.random.default_rng(self.config.seed)

        self.true_preferences: np.ndarray = np.array([])  # [n_agents, n_topics]
        self.observed_preferences: np.ndarray = np.array([])  # Algorithm's model
        self.content_pool: np.ndarray = np.array([])  # [n_content, n_topics]
        self.round_data: list[dict] = []

    def run(self) -> SimulationResult:
        """Run the recommendation simulation."""
        self._initialize()
        performativity_scores = []
        initial_preferences = self.true_preferences.copy()

        for round_idx in range(self.config.n_rounds):
            round_info = self._run_round(round_idx)
            performativity_scores.append(round_info["performativity"])
            self.round_data.append(round_info)

        # Measure how much true preferences drifted toward recommendations
        preference_drift = float(np.mean(
            np.linalg.norm(self.true_preferences - initial_preferences, axis=1)
        ))

        # Measure filter bubble: decrease in preference diversity
        initial_diversity = float(np.mean(np.std(initial_preferences, axis=1)))
        final_diversity = float(np.mean(np.std(self.true_preferences, axis=1)))
        bubble_strength = max(0.0, initial_diversity - final_diversity) / max(initial_diversity, 1e-10)

        return SimulationResult(
            scenario="content_recommendation",
            n_agents=self.config.n_agents,
            n_rounds=self.config.n_rounds,
            performativity_scores=performativity_scores,
            accuracy_naive=float(np.mean([r["engagement_rate"] for r in self.round_data])),
            accuracy_corrected=float(np.mean([r["true_match"] for r in self.round_data])),
            bias_introduced=preference_drift,
            details={
                "preference_drift": preference_drift,
                "initial_diversity": initial_diversity,
                "final_diversity": final_diversity,
                "filter_bubble_strength": bubble_strength,
                "avg_engagement_rate": float(
                    np.mean([r["engagement_rate"] for r in self.round_data])
                ),
            },
        )

    def _initialize(self) -> None:
        """Initialize users, preferences, and content pool."""
        n = self.config.n_agents

        # Users have diverse initial preferences (Dirichlet distribution)
        self.true_preferences = self.rng.dirichlet(
            alpha=np.ones(self.n_topics) * 2.0, size=n
        )
        self.observed_preferences = self.true_preferences.copy()

        # Content pool: each item has a topic distribution
        n_content = max(self.n_topics * 10, 100)
        self.content_pool = self.rng.dirichlet(
            alpha=np.ones(self.n_topics) * 0.5, size=n_content
        )

    def _run_round(self, round_idx: int) -> dict:
        """Run a single recommendation round."""
        n = self.config.n_agents
        influence = self.config.prediction_influence

        # Step 1: Algorithm predicts preferences and recommends content
        recommendations = self._recommend(self.observed_preferences)

        # Step 2: Users engage based on true preferences + exposure effect
        engagements = np.zeros(n)
        for i in range(n):
            # True affinity for recommended content
            true_affinity = float(np.dot(self.true_preferences[i], recommendations[i]))
            # Exposure effect: mere exposure increases engagement
            exposure_boost = influence * 0.2
            engage_prob = np.clip(true_affinity + exposure_boost, 0.0, 1.0)
            engagements[i] = float(self.rng.random() < engage_prob)

        # Step 3: Update observed preferences based on engagement
        for i in range(n):
            if engagements[i] > 0.5:
                # Algorithm sees engagement -> reinforces prediction
                self.observed_preferences[i] = (
                    0.9 * self.observed_preferences[i] + 0.1 * recommendations[i]
                )
                self.observed_preferences[i] /= self.observed_preferences[i].sum()

        # Step 4: TRUE preference drift due to exposure (the performative effect)
        for i in range(n):
            if engagements[i] > 0.5:
                drift = influence * 0.05 * (recommendations[i] - self.true_preferences[i])
                self.true_preferences[i] += drift
                self.true_preferences[i] = np.clip(self.true_preferences[i], 0.01, None)
                self.true_preferences[i] /= self.true_preferences[i].sum()

        # Compute performativity score
        true_match = float(np.mean([
            np.dot(self.true_preferences[i], recommendations[i])
            for i in range(n)
        ]))

        # Observed match (what algorithm sees)
        observed_match = float(np.mean(engagements))

        performativity = abs(observed_match - true_match)

        return {
            "round": round_idx,
            "engagement_rate": observed_match,
            "true_match": true_match,
            "performativity": performativity,
            "preference_entropy": float(np.mean([
                -np.sum(p * np.log(p + 1e-10)) for p in self.true_preferences
            ])),
        }

    def _recommend(self, preferences: np.ndarray) -> np.ndarray:
        """Recommend content to each user based on predicted preferences.

        Returns the topic distribution of recommended content for each user.
        """
        n = len(preferences)
        recommendations = np.zeros((n, self.n_topics))

        for i in range(n):
            # Score all content by predicted preference match
            scores = self.content_pool @ preferences[i]
            # Recommend top content (with some exploration noise)
            noise = self.rng.normal(0, 0.01, size=len(scores))
            best_idx = np.argmax(scores + noise)
            recommendations[i] = self.content_pool[best_idx]

        return recommendations
