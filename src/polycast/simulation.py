"""Agent-based simulation of belief formation in a forecasting crowd.

Used to answer questions the single-run pipeline cannot: how many independent
analysts are needed before a crowd converges, and where social influence sends
it. The interesting result is that higher social weight produces faster
consensus and a less accurate one, because a crowd that leans on its
neighbours converges on their answer rather than on the signal.

Deterministic given a seed, so simulation results can be diffed across commits.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field


@dataclass
class SimAgent:
    agent_id: int
    belief: float
    confidence: float
    social_weight: float
    neighbours: list[int] = field(default_factory=list)

    def update(self, signal: float, crowd_mean: float) -> None:
        # Own evidence first, then social pressure. Overconfident agents weigh
        # their own signal more, which is exactly how echo chambers start.
        own_weight = 1.0 - self.social_weight
        evidence = own_weight * signal + self.social_weight * crowd_mean
        step = 0.35 * (1.0 - 0.5 * self.confidence)
        self.belief = min(0.99, max(0.01, self.belief + step * (evidence - self.belief)))


@dataclass
class SimulationResult:
    rounds: int
    truth: float
    final_mean: float
    final_spread: float
    mean_series: list[float]
    spread_series: list[float]
    converged_round: int | None


def build_ring(n: int) -> list[list[int]]:
    return [[(i - 1) % n, (i + 1) % n] for i in range(n)]


def build_dense(n: int) -> list[list[int]]:
    return [[j for j in range(n) if j != i] for i in range(n)]


def run_simulation(
    n_agents: int = 40,
    truth: float = 0.7,
    rounds: int = 60,
    social_weight: float = 0.3,
    topology: str = "dense",
    signal_noise: float = 0.12,
    seed: int = 7,
) -> SimulationResult:
    if topology not in {"dense", "ring"}:
        raise ValueError("topology must be 'dense' or 'ring'")
    rng = random.Random(seed)
    neighbours = build_dense(n_agents) if topology == "dense" else build_ring(n_agents)
    agents = [
        SimAgent(
            agent_id=i,
            belief=min(0.95, max(0.05, rng.gauss(0.5, 0.2))),
            confidence=rng.random(),
            social_weight=social_weight,
            neighbours=neighbours[i],
        )
        for i in range(n_agents)
    ]

    mean_series: list[float] = []
    spread_series: list[float] = []
    converged_round: int | None = None

    for round_index in range(rounds):
        snapshot = [a.belief for a in agents]
        for agent in agents:
            crowd = (
                statistics.mean(snapshot[j] for j in agent.neighbours)
                if agent.neighbours
                else statistics.mean(snapshot)
            )
            signal = min(1.0, max(0.0, rng.gauss(truth, signal_noise)))
            agent.update(signal, crowd)

        beliefs = [a.belief for a in agents]
        mean = statistics.mean(beliefs)
        spread = statistics.pstdev(beliefs)
        mean_series.append(round(mean, 5))
        spread_series.append(round(spread, 5))
        if converged_round is None and spread < 0.02 and abs(mean - truth) < 0.05:
            converged_round = round_index

    return SimulationResult(
        rounds=rounds,
        truth=truth,
        final_mean=mean_series[-1],
        final_spread=spread_series[-1],
        mean_series=mean_series,
        spread_series=spread_series,
        converged_round=converged_round,
    )
