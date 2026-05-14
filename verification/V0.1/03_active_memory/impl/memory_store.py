"""Active Memory — dual-layer (STM/LTM) with homeostasis-coupled forgetting.

STM: short-term, fast decay, small capacity (~200 items)
LTM: long-term, slow decay, only important events enter

The agent can "feel" memory load via memory_load_drive, which feeds back
into homeostasis and accelerates forgetting when memory is full.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MemoryItem:
    id: int
    step: int
    pos: tuple[int, int]
    event_type: str      # "food_found" | "danger" | "new_area" | "food_visible"
    energy: float        # homeostasis energy at time of event
    novelty: float       # homeostasis novelty
    safety: float         # homeostasis safety
    importance: float     # 0..1, computed at encoding time
    decay: float = 1.0    # starts at 1.0, decays toward 0 over time
    access_count: int = 0 # how many times retrieved


class ActiveMemory:
    def __init__(self, stm_capacity: int = 200, ltm_capacity: int = 500,
                 stm_decay: float = 0.005, ltm_decay: float = 0.001,
                 ltm_importance_threshold: float = 0.6):
        self.stm: list[MemoryItem] = []
        self.ltm: list[MemoryItem] = []
        self.stm_capacity = stm_capacity
        self.ltm_capacity = ltm_capacity
        self.stm_decay = stm_decay
        self.ltm_decay = ltm_decay
        self.ltm_threshold = ltm_importance_threshold
        self._next_id = 0

    def write(self, item: MemoryItem):
        item.id = self._next_id; self._next_id += 1
        self.stm.append(item)
        self._evict(self.stm, self.stm_capacity)

        # Promote to LTM if important enough
        if item.importance >= self.ltm_threshold:
            self.ltm.append(item)
            self._evict(self.ltm, self.ltm_capacity)

    def decay_step(self, memory_load_drive: float = 0.0):
        """Decay all items. Higher memory_load → faster decay."""
        rate_stm = self.stm_decay * (1.0 + memory_load_drive * 2.0)
        rate_ltm = self.ltm_decay * (1.0 + memory_load_drive)

        for items, rate in [(self.stm, rate_stm), (self.ltm, rate_ltm)]:
            for item in items:
                item.decay = max(0.0, item.decay - rate)
            # Remove decayed items
            items[:] = [i for i in items if i.decay > 0.05]

    def retrieve(self, pos: tuple[int, int], k: int = 3) -> list[MemoryItem]:
        seen = set()
        candidates = []
        for item in self.stm + self.ltm:
            if item.id in seen: continue
            seen.add(item.id)
            pos_match = 1.0 if item.pos == pos else (
                0.3 if self._distance(item.pos, pos) <= 2 else 0.0
            )
            score = item.importance * item.decay * (0.5 + 0.5 * pos_match)
            candidates.append((score, item))

        candidates.sort(key=lambda x: x[0], reverse=True)
        result = [it for _, it in candidates[:k]]

        for item in result:
            item.access_count += 1
        return result

    def get_memory_load(self) -> float:
        """0..1: how full is STM?"""
        return min(1.0, len(self.stm) / max(1, self.stm_capacity))

    def _evict(self, items, capacity):
        """Remove least important + most decayed items."""
        if len(items) <= capacity:
            return
        items.sort(key=lambda i: i.importance * i.decay)
        items[:] = items[-(capacity):]

    def _distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
