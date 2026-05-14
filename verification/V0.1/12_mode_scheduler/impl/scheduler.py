"""Mode scheduler — determines agent mode from homeostasis state.

Uses hysteresis to prevent rapid mode switching:
  Enter HOT when energy < 0.3 or safety > 0.5
  Exit  HOT when energy > 0.5 and safety < 0.3
  Enter COLD when energy > 0.95 (fully satisfied, no drives)

Call every step to get the current mode.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "06_hot_warm_cold", "impl"))

from modes import AgentMode


class ModeScheduler:
    def __init__(self):
        self.current = AgentMode.HOT

    def update(self, energy: float, novelty: float, safety: float) -> AgentMode:
        # Safety overrides everything
        if safety > 0.5:
            self.current = AgentMode.HOT
            return self.current

        # Energy-driven transitions with hysteresis
        if self.current == AgentMode.HOT:
            if safety < 0.3 and energy > 0.5 and novelty < 0.9:
                self.current = AgentMode.WARM
            elif energy > 0.95:
                self.current = AgentMode.COLD
        elif self.current == AgentMode.WARM:
            if safety > 0.5 or energy < 0.3:
                self.current = AgentMode.HOT
            elif energy > 0.95:
                self.current = AgentMode.COLD
        elif self.current == AgentMode.COLD:
            if safety > 0.5 or energy < 0.5 or novelty > 0.5:
                self.current = AgentMode.HOT

        return self.current
