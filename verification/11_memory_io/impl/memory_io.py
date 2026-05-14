"""Memory I/O — encodes perception events and retrieves relevant memories.

Bridges 10_sensorimotor (perception) → 03_active_memory (storage).
"""

import os, sys
_IMPL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_IMPL, "..", "..", "03_active_memory", "impl"))

from memory_store import ActiveMemory, MemoryItem


class MemoryIO:
    def __init__(self, memory: ActiveMemory):
        self.memory = memory

    def encode(self, step: int, pos: tuple[int, int],
               energy: float, novelty: float, safety: float,
               food_eaten: bool, damage_taken: bool, new_tile: bool,
               food_visible: bool):
        """Encode a perception event into a MemoryItem."""
        # Compute importance
        importance = 0.1  # baseline
        if food_eaten:
            importance = 1.0
            event = "food_found"
        elif damage_taken:
            importance = 0.9
            event = "danger"
        elif new_tile:
            importance = 0.5
            event = "new_area"
        elif food_visible:
            importance = 0.3
            event = "food_visible"
        else:
            return  # not worth remembering

        item = MemoryItem(
            id=0, step=step, pos=pos, event_type=event,
            energy=energy, novelty=novelty, safety=safety,
            importance=importance,
        )
        self.memory.write(item)

    def retrieve_context(self, pos, k: int = 3) -> list[float]:
        """Retrieve memory context as a fixed-size float vector.
        If pos is None, returns global averages.
        """
        if pos is None:
            return [0.0, 0.0, 0.0, 0.0]
        items = self.memory.retrieve(pos, k)
        if not items:
            return [0.0, 0.0, 0.0, 0.0]

        food_hints = sum(1 for i in items if i.event_type in ("food_found", "food_visible"))
        danger_hints = sum(1 for i in items if i.event_type == "danger")
        avg_imp = sum(i.importance for i in items) / len(items)
        avg_decay = sum(i.decay for i in items) / len(items)

        return [
            food_hints / len(items),
            danger_hints / len(items),
            avg_imp,
            avg_decay,
        ]
