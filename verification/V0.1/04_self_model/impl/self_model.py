"""Self Model — the agent's internal representation of itself.

Tracks: homeostasis levels, recent behavior statistics, meta-state.
Updates via dual-speed EMA (slow for identity, fast for current mood).
Encodes into a compact self_vector for injection into LTC decision-making.

The self_vector answers: "If I am me, what should I do?"
"""

from dataclasses import dataclass, field
from collections import deque
import numpy as np


@dataclass
class SelfState:
    """Raw self-perception at a given moment."""
    energy: float = 0.8
    novelty: float = 1.0
    safety: float = 0.0
    recent_eat_rate: float = 0.0       # fraction of recent steps with food found
    recent_damage_rate: float = 0.0    # fraction of recent steps with damage
    exploration_rate: float = 0.0      # fraction of recent steps on new tiles
    steps_since_event: int = 0         # steps since last meaningful event
    drive_stability: float = 0.0       # inverse variance of drive (0=unstable, 1=stable)


class SelfModel:
    """Maintains and encodes the agent's self-state.

    Dual-speed EMA:
      slow (α=0.01): "who I am" — long-term identity
      fast (α=0.15): "how I feel now" — short-term mood
    """

    def __init__(self, slow_alpha: float = 0.01, fast_alpha: float = 0.15,
                 recent_window: int = 50, embed_dim: int = 8):
        self.slow_alpha = slow_alpha
        self.fast_alpha = fast_alpha
        self.recent_window = recent_window
        self.embed_dim = embed_dim

        # State
        self.slow = np.zeros(6, dtype=np.float32)
        self.fast = np.zeros(6, dtype=np.float32)

        # Rolling windows for behavior statistics
        self._eat_history: deque[bool] = deque(maxlen=recent_window)
        self._damage_history: deque[bool] = deque(maxlen=recent_window)
        self._new_tile_history: deque[bool] = deque(maxlen=recent_window)
        self._drive_history: deque[np.ndarray] = deque(maxlen=recent_window)
        self._steps_since_event = 0
        self._initialized = False

    def update(self, energy: float, novelty: float, safety: float,
               food_eaten: bool, damage_taken: bool, new_tile: bool):
        """Called every step with current state. Updates rolling stats and EMA."""
        raw = np.array([energy, novelty, safety,
                        self._mean(self._eat_history),
                        self._mean(self._damage_history),
                        self._mean(self._new_tile_history)], dtype=np.float32)

        self._eat_history.append(food_eaten)
        self._damage_history.append(damage_taken)
        self._new_tile_history.append(new_tile)

        drive = np.array([1.0 - energy, max(0.0, 1.0 - self._mean(self._eat_history)),
                          safety], dtype=np.float32)
        self._drive_history.append(drive)

        if food_eaten or damage_taken or new_tile:
            self._steps_since_event = 0
        else:
            self._steps_since_event += 1

        if not self._initialized:
            self.slow = raw.copy()
            self.fast = raw.copy()
            self._initialized = True
        else:
            self.slow += self.slow_alpha * (raw - self.slow)
            self.fast += self.fast_alpha * (raw - self.fast)

    def encode(self) -> np.ndarray:
        """Return the self_vector (concatenated slow + selected fast dims)."""
        if not self._initialized:
            return np.zeros(self.embed_dim, dtype=np.float32)

        drive_stability = self._compute_stability()
        event_freshness = min(1.0, self._steps_since_event / 200.0)

        vec = np.concatenate([
            self.slow[:3],             # long-term: energy, novelty, safety
            self.fast[3:6],            # short-term: behavior rates
            [drive_stability],         # meta: how stable
            [event_freshness],         # meta: time since last event
        ])  # 8-dim

        return vec.astype(np.float32)

    def get_state(self) -> SelfState:
        return SelfState(
            energy=self.slow[0], novelty=self.slow[1], safety=self.slow[2],
            recent_eat_rate=self._mean(self._eat_history),
            recent_damage_rate=self._mean(self._damage_history),
            exploration_rate=self._mean(self._new_tile_history),
            steps_since_event=self._steps_since_event,
            drive_stability=self._compute_stability(),
        )

    def _compute_stability(self) -> float:
        if len(self._drive_history) < 4:
            return 0.5
        stack = np.stack(list(self._drive_history), axis=0)
        var = np.var(stack, axis=0).mean()
        return float(1.0 / (1.0 + var))

    def _mean(self, dq: deque) -> float:
        if not dq:
            return 0.0
        return sum(dq) / len(dq)
