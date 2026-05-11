"""Actuator — translates BehaviorPrimitive → concrete Action.

The actuator knows how to "execute" a primitive given the current facts
(e.g. APPROACH food → MOVE toward the nearest food direction).
"""

from behavior_standard.primitives import BehaviorPrimitive


class Actuator:
    """Translates a BehaviorPrimitive into a concrete action index (0-5).

    The actuator receives the selected primitive and the facts,
    then chooses the specific motor command.
    """

    def __init__(self, rng=None):
        import numpy as np
        self._rng = rng or np.random.default_rng()

    def execute(self, primitive: BehaviorPrimitive, facts) -> int:
        """Returns an Action integer (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=EAT, 5=WAIT)."""

        if primitive == BehaviorPrimitive.CONSUME:
            return 4  # EAT

        if primitive == BehaviorPrimitive.WAIT:
            return 5  # WAIT

        if primitive == BehaviorPrimitive.APPROACH:
            return self._dir_to_action(
                facts.food_dirs[0] if facts.food_dirs else self._random_dir()
            )

        if primitive == BehaviorPrimitive.AVOID:
            if facts.danger_dirs:
                return self._dir_to_action(self._opposite(facts.danger_dirs[0]))
            return self._random_action()

        if primitive == BehaviorPrimitive.EXPLORE:
            dirs = ["UP", "DOWN", "LEFT", "RIGHT"]
            return self._dir_to_action(dirs[self._rng.integers(0, 4)])

        if primitive == BehaviorPrimitive.WANDER:
            return self._random_action()

        return 5  # fallback WAIT

    def _dir_to_action(self, d: str) -> int:
        return {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}.get(d, 5)

    def _opposite(self, d: str) -> str:
        return {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}.get(d, "UP")

    def _random_dir(self) -> str:
        return ["UP", "DOWN", "LEFT", "RIGHT"][self._rng.integers(0, 4)]

    def _random_action(self) -> int:
        return self._rng.integers(0, 6)
