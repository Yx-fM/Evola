"""Behavior primitives — irreducible atomic behaviors.

These are the "motors" that all complex behaviors decompose into.
The primitive set can grow per intelligence level but never loses primitives.
"""

from enum import Enum


class BehaviorPrimitive(Enum):
    APPROACH = "approach"    # move toward a target (food, interesting area)
    AVOID = "avoid"          # move away from a target (danger)
    CONSUME = "consume"      # act on current cell (eat, collect)
    WAIT = "wait"            # do nothing this step
    EXPLORE = "explore"      # move toward unvisited area
    WANDER = "wander"        # random movement with no target
