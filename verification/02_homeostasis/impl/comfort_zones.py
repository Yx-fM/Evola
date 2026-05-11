"""Comfort zones — the 'set-points' for each homeostatic variable.

A comfort zone is an interval [low, high]. The variable is 'comfortable'
when its value falls within this interval. Deviation outside produces a drive.

Key design choice: comfort is a ZONE, not a point. The organism is not trying
to reach exactly 0.7 energy — anywhere in [0.5, 1.0] is fine.
"""

from dataclasses import dataclass


@dataclass
class ComfortZone:
    low: float
    high: float

    def deviation(self, value: float) -> float:
        """How far outside the comfort zone? 0 = inside, >0 = deviation."""
        if self.low <= value <= self.high:
            return 0.0
        if value < self.low:
            return self.low - value
        return value - self.high

    def is_comfortable(self, value: float) -> bool:
        return self.low <= value <= self.high


# Default zones — tuned for a "fish-level" agent in 2D grid world
ENERGY_ZONE   = ComfortZone(low=0.5, high=1.0)   # drive below 0.5
NOVELTY_ZONE  = ComfortZone(low=0.3, high=0.8)    # drive both too-low and too-high
SAFETY_ZONE   = ComfortZone(low=0.0, high=0.2)    # only drive when danger significant
