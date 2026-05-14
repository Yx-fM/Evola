"""Evola IQ — 智能水平六维评价系统

Evaluates agent intelligence across 6 dimensions:
  S: Survival   (存活)  — energy stability, time in comfort zone
  E: Exploration (探索) — visited ratio, novelty change
  L: Learning   (学习)  — efficiency improvement over time
  A: Adaptation (适应)  — recovery speed after damage
  F: Efficiency (效率)  — food eaten per step
  T: Stability  (稳定)  — overall homeostasis health

Weighted score: S*0.25 + E*0.15 + L*0.20 + A*0.10 + F*0.15 + T*0.15

Usage:
    iq = EvolaIQ()
    iq.record(energy, novelty, safety, food_eaten, damage_taken, visited_count, total_cells)
    score = iq.evaluate()  # → dict with all 6 dimensions + composite score
"""

from dataclasses import dataclass, field
from collections import deque
import numpy as np


@dataclass
class IQResult:
    survival: float = 0.0
    exploration: float = 0.0
    learning: float = 0.0
    adaptation: float = 0.0
    efficiency: float = 0.0
    stability: float = 0.0
    composite: float = 0.0

    def to_dict(self):
        return {
            "survival": self.survival, "exploration": self.exploration,
            "learning": self.learning, "adaptation": self.adaptation,
            "efficiency": self.efficiency, "stability": self.stability,
            "composite": self.composite,
        }


class EvolaIQ:
    def __init__(self, window_size: int = 200):
        self.window = window_size
        self.total_steps = 0
        self.food_eaten = 0
        self.damage_events = 0
        self.steps_since_damage = 0
        self.damage_recovery_steps: list[int] = []
        self.energy_history: deque[float] = deque(maxlen=window_size)
        self.novelty_history: deque[float] = deque(maxlen=window_size)
        self.safety_history: deque[float] = deque(maxlen=window_size)
        self.comfort_time = 0
        self.early_phase = deque(maxlen=100)  # first 100 steps food efficiency
        self.late_phase = deque(maxlen=100)   # last 100 steps food efficiency
        self.visited_history: deque[int] = deque(maxlen=50)

    def record(self, energy: float, novelty: float, safety: float,
               food_eaten: bool, damage_taken: bool,
               visited_count: int, total_cells: int):
        self.total_steps += 1
        self.energy_history.append(energy)
        self.novelty_history.append(novelty)
        self.safety_history.append(safety)
        self.visited_history.append(visited_count)

        if food_eaten:
            self.food_eaten += 1
            if self.total_steps <= 100:
                self.early_phase.append(self.total_steps)

        if damage_taken:
            self.damage_events += 1
            if self.steps_since_damage > 0:
                self.damage_recovery_steps.append(self.steps_since_damage)
            self.steps_since_damage = 0
        else:
            self.steps_since_damage += 1

        # Comfort zone check
        if self._in_comfort(energy, novelty, safety):
            self.comfort_time += 1

        # Late phase tracking
        if self.total_steps > 100:
            self.late_phase.append(self.total_steps)

    def evaluate(self) -> IQResult:
        r = IQResult()
        if self.total_steps < 20:
            return r

        r.survival = self._score_survival()
        r.exploration = self._score_exploration()
        r.learning = self._score_learning()
        r.adaptation = self._score_adaptation()
        r.efficiency = self._score_efficiency()
        r.stability = self._score_stability()
        r.composite = (
            r.survival * 0.25 + r.exploration * 0.15 + r.learning * 0.20 +
            r.adaptation * 0.10 + r.efficiency * 0.15 + r.stability * 0.15
        )
        return r

    def _score_survival(self) -> float:
        """Energy stability: inverse of variance + comfort zone ratio."""
        if len(self.energy_history) < 10:
            return 0.5
        var = np.var(list(self.energy_history))
        stability = 1.0 / (1.0 + 5.0 * var)
        comfort_ratio = self.comfort_time / max(1, self.total_steps)
        return (stability * 0.6 + comfort_ratio * 0.4)

    def _score_exploration(self) -> float:
        """How much of the world has been explored."""
        if len(self.visited_history) < 5:
            return 0.0
        # Use recent trend
        recent = list(self.visited_history)[-10:]
        growth = (recent[-1] - recent[0]) / max(1, len(recent))
        return min(1.0, max(0.0, growth / 3.0))

    def _score_learning(self) -> float:
        """Efficiency improvement: late phase vs early phase."""
        if len(self.early_phase) < 5 or len(self.late_phase) < 5:
            return 0.3  # neutral
        # Steps per food: lower is better
        early_rate = len(self.early_phase) / min(100, self.early_phase[-1])
        late_steps = min(len(self.late_phase), self.late_phase[-1] - 100)
        late_rate = max(0.001, len(self.late_phase) / max(1, late_steps))
        improvement = late_rate / max(0.001, early_rate)
        return min(1.0, max(0.0, (improvement - 0.5)))

    def _score_adaptation(self) -> float:
        """Recovery speed after damage."""
        if not self.damage_recovery_steps:
            return 0.5  # no damage events, neutral
        avg_recovery = np.mean(self.damage_recovery_steps)
        # Faster recovery = higher score (0 steps = best, >50 = poor)
        return max(0.0, 1.0 - avg_recovery / 50.0)

    def _score_efficiency(self) -> float:
        """Food eaten per 100 steps."""
        rate = self.food_eaten / max(1, self.total_steps) * 100
        return min(1.0, max(0.0, rate / 5.0))

    def _score_stability(self) -> float:
        """Overall homeostasis health."""
        if len(self.energy_history) < 10:
            return 0.5
        ev = np.mean(list(self.energy_history))
        sv = np.mean(list(self.safety_history))
        # High energy + low safety = stable
        return (ev * 0.7 + (1.0 - sv) * 0.3)

    def _in_comfort(self, energy, novelty, safety):
        return (0.4 < energy < 1.0 and 0.2 < novelty < 0.9 and safety < 0.5)
