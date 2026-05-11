"""Agent interface: what every intelligent agent loaded into Kunyu must implement.

The world (08) provides observations and accepts actions.
The agent (independent module) carries its own memory, drives, and internal state.
The agent's "mind" is never stored in the world — it's hers alone.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class IAgent(ABC):
    """Any agent that can live in Kunyu World.

    The world calls act(obs) → action, then step(action), then provides
    the outcome via observe(info). The agent's internal state (memory,
    homeostasis, self-model) lives here, NOT in the world.
    """

    @property
    @abstractmethod
    def agent_id(self) -> str: ...

    @abstractmethod
    def act(self, obs: np.ndarray) -> int:
        """Given 5x5 observation, return an Action (int)."""
        ...

    @abstractmethod
    def observe(self, info: dict) -> None:
        """Receive the outcome of the last action.
        info contains: energy_gained, damage_taken, collision, etc.
        The agent updates her internal state (homeostasis, memory)."""
        ...

    def save_mind(self, path: str) -> None:
        """Serialize the agent's internal state to disk (optional)."""
        pass

    def load_mind(self, path: str) -> None:
        """Restore the agent's internal state from disk (optional)."""
        pass


class RandomAgent(IAgent):
    """Baseline: picks a random action every step. No memory."""

    def __init__(self, agent_id: str = "random", rng: Optional[np.random.Generator] = None):
        self._id = agent_id
        self._rng = rng or np.random.default_rng()
        self.step_count = 0
        self.last_info = {}

    @property
    def agent_id(self) -> str:
        return self._id

    def act(self, obs: np.ndarray) -> int:
        self.step_count += 1
        return self._rng.integers(0, 6)

    def observe(self, info: dict) -> None:
        self.last_info = info
