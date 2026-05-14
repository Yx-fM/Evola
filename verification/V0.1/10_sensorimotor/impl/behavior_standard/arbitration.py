"""Arbitration layer — maps drive vectors + context to behavior primitives.

P0: RuleArbiter  (dynamic weight decay, hand-crafted rules)
P1: LTCArbiter   (liquid time-constant network, same interface)

The arbiter IS the "decision" — the core of "what should I do right now?"
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import math

from behavior_standard.primitives import BehaviorPrimitive


@dataclass
class DriveVector:
    energy: float = 0.0    # 0=sated, 1=starving
    novelty: float = 0.0   # 0=familiar, 1=bored
    safety: float = 0.0    # 0=safe, 1=in danger


@dataclass
class Facts:
    """What the agent perceives from the environment."""
    food_dirs: list[str] = field(default_factory=list)        # ["UP", "LEFT", ...]
    danger_dirs: list[str] = field(default_factory=list)      # ["DOWN", ...]
    is_new_tile: bool = False
    n_walls_near: int = 0
    n_open_cells: int = 0


@dataclass
class SalienceMap:
    """How 'important' each fact is given the current drive state."""
    food_relevance: float = 0.0
    danger_relevance: float = 0.0
    novelty_relevance: float = 0.0


@dataclass
class DecisionContext:
    """Additional context the arbiter can use (past action, memory, self-model)."""
    last_action: Optional[str] = None
    same_action_streak: int = 0


class IArbiter(ABC):
    """Permanent interface: drive + facts + salience + context → primitive."""

    @abstractmethod
    def select(self, drive: DriveVector, facts: Facts,
               salience: SalienceMap, context: DecisionContext) -> BehaviorPrimitive:
        ...


class RuleArbiter(IArbiter):
    """P0: Rule-based arbiter with dynamic weight scheduling (not binary masks).

    Priority order (soft):
      1. Survive — if danger present and safety_drive > threshold, AVOID
      2. Sustain — if hungry and food visible, APPROACH / CONSUME
      3. Explore — if safe and novelty_drive high, EXPLORE
      4. Default — WANDER
    """

    def __init__(self, safety_threshold: float = 0.3,
                 energy_urgent: float = 0.5,
                 energy_safe: float = 0.3,
                 novelty_threshold: float = 0.2):
        self.safety_threshold = safety_threshold
        self.energy_urgent = energy_urgent
        self.energy_safe = energy_safe
        self.novelty_threshold = novelty_threshold

    def select(self, drive: DriveVector, facts: Facts,
               salience: SalienceMap, context: DecisionContext) -> BehaviorPrimitive:

        # Layer 0: Safety — dynamic weight decay on lower priorities
        safety_mod = self._safety_mod(drive.safety)

        if drive.safety > self.safety_threshold and facts.danger_dirs:
            return BehaviorPrimitive.AVOID

        # Layer 1: Survival — hunger drives approach/consume
        if drive.energy > self.energy_urgent and facts.food_dirs:
            if "HERE" in facts.food_dirs:
                return BehaviorPrimitive.CONSUME
            return BehaviorPrimitive.APPROACH

        # Layer 2: Exploration — soft suppression by hunger
        if drive.energy < self.energy_safe and drive.novelty * safety_mod > self.novelty_threshold:
            return BehaviorPrimitive.EXPLORE

        # Layer 3: Default wandering
        return BehaviorPrimitive.WANDER

    def _safety_mod(self, safety_drive: float) -> float:
        if safety_drive <= self.safety_threshold:
            return 1.0
        r = (safety_drive - self.safety_threshold) / (1.0 - self.safety_threshold)
        return max(0.05, 1.0 - r)
