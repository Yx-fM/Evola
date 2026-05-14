"""Homeostatic variables — each variable has a current value and update logic.

Three innate variables:
- energy: depleted over time, restored by eating
- novelty: based on how much of the map has been explored
- safety: based on how close/frequent danger encounters are

All values are clipped to [0, 1].
"""

import numpy as np


class EnergyVariable:
    """Depletes over time. Eating restores it."""

    def __init__(self, initial: float = 0.8, decay_rate: float = 0.003):
        self.value = float(np.clip(initial, 0.0, 1.0))
        self.decay_rate = decay_rate

    def step(self, dt: float = 1.0):
        self.value = max(0.0, self.value - self.decay_rate * dt)

    def replenish(self, amount: float):
        self.value = min(1.0, self.value + amount)

    def deplete(self, amount: float):
        self.value = max(0.0, self.value - amount)


class NoveltyVariable:
    """Tracks how much of the environment has been explored. Low visited ratio = high novelty."""

    def __init__(self, total_cells: int = 300):
        self.total_cells = max(1, total_cells)
        self.visited_count = 0

    @property
    def value(self) -> float:
        explored_ratio = self.visited_count / self.total_cells
        return max(0.0, 1.0 - explored_ratio)

    def visit(self):
        self.visited_count += 1

    def set_visited(self, count: int):
        self.visited_count = min(self.total_cells, max(0, count))


class SafetyVariable:
    """Tracks recent danger exposure via EMA. Fast to rise, slow to decay."""

    def __init__(self, rise_rate: float = 0.8, decay_rate: float = 0.05):
        self.value = 0.0
        self.rise_rate = rise_rate
        self.decay_rate = decay_rate

    def step(self, danger_present: bool, dt: float = 1.0):
        if danger_present:
            self.value += (1.0 - self.value) * self.rise_rate * dt
        else:
            self.value *= (1.0 - self.decay_rate * dt)
        self.value = float(np.clip(self.value, 0.0, 1.0))
