"""Homeostasis — the core of Evola's 'needs' system.

Takes environmental facts, updates internal variables, produces drive vectors.
The drive vector tells the sensorimotor system: 'this is what I need right now.'
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from comfort_zones import ENERGY_ZONE, NOVELTY_ZONE, SAFETY_ZONE
from variables import EnergyVariable, NoveltyVariable, SafetyVariable


@dataclass
class DriveVector:
    """The agent's current motivational state. All values in [0, 1]."""
    energy: float = 0.0
    novelty: float = 0.0
    safety: float = 0.0

    def as_dict(self) -> dict:
        return {"energy": self.energy, "novelty": self.novelty, "safety": self.safety}

    def labels(self) -> dict:
        """Human-readable labels for JunJian's PUBLIC privacy level."""
        labels = {}
        labels["energy"] = _label(self.energy, [
            (0.2, "starving"), (0.5, "hungry"), (0.8, "peckish"), (1.0, "full")
        ])
        labels["novelty"] = _label(self.novelty, [
            (0.2, "familiar"), (0.5, "curious"), (0.8, "restless"), (1.0, "bored")
        ])
        labels["safety"] = _label(self.safety, [
            (0.2, "safe"), (0.5, "alert"), (0.8, "scared"), (1.0, "terrified")
        ])
        return labels


def _label(value: float, thresholds: list[tuple[float, str]]) -> str:
    for threshold, name in thresholds:
        if value <= threshold:
            return name
    return thresholds[-1][1] if thresholds else "unknown"


class Homeostasis:
    """One Homeostasis instance per agent. The agent's 'body' that tells her brain what she needs."""

    def __init__(self, total_cells: int = 300,
                 energy_decay: float = 0.003,
                 safety_rise: float = 0.8,
                 safety_decay: float = 0.05):
        self.energy = EnergyVariable(initial=0.8, decay_rate=energy_decay)
        self.novelty = NoveltyVariable(total_cells=total_cells)
        self.safety = SafetyVariable(rise_rate=safety_rise, decay_rate=safety_decay)
        self.step_count = 0

    def update(self, info: dict, dt: float = 1.0):
        """Called every world step. info comes from KunyuWorld.step()."""
        self.step_count += 1

        # Energy: always decay, restore if ate food
        self.energy.step(dt)
        gained = info.get("energy_gained", 0.0)
        if gained > 0:
            self.energy.replenish(gained)
        damage = info.get("damage_taken", 0.0)
        if damage > 0:
            self.energy.deplete(damage * 2.0)

        # Novelty: track how many cells visited
        agent_pos = info.get("agent_pos")
        if agent_pos is not None:
            self.novelty.visit()

        # Safety: update based on whether danger is nearby
        danger_visible = info.get("danger_visible", False)
        self.safety.step(danger_visible, dt)

    def get_drive(self) -> DriveVector:
        """Compute current drive vector from comfort zone deviations."""
        return DriveVector(
            energy=ENERGY_ZONE.deviation(self.energy.value),
            novelty=NOVELTY_ZONE.deviation(self.novelty.value),
            safety=SAFETY_ZONE.deviation(self.safety.value),
        )

    def get_state(self) -> dict:
        drive = self.get_drive()
        return {
            "energy": self.energy.value,
            "novelty": self.novelty.value,
            "safety": self.safety.value,
            "drive": drive.as_dict(),
            "labels": drive.labels(),
        }
