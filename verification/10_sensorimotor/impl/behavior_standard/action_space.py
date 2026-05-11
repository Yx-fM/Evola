"""Action space — maps behavior primitives to concrete Actions per intelligence level.

Higher intelligence adds new Actions; lower levels use only their subset.
The framework (primitives, arbiter interface) is permanent.
"""

from enum import Enum


class IntelligenceLevel(Enum):
    FISH = "fish"       # P0: basic survival
    CAT = "cat"         # P2: simple tool use
    DOG = "dog"         # P2: social
    MONKEY = "monkey"   # future: complex tools
    HUMAN = "human"     # future


# Actions per level (will expand as we add more capability types)
# Currently all levels share the same 6 actions from 08_kunyu:entities.Action
# In the future, higher levels get additional actions.
LEVEL_ACTIONS = {
    IntelligenceLevel.FISH: ["UP", "DOWN", "LEFT", "RIGHT", "EAT", "WAIT"],
    IntelligenceLevel.CAT: ["UP", "DOWN", "LEFT", "RIGHT", "EAT", "WAIT"],
    IntelligenceLevel.DOG: ["UP", "DOWN", "LEFT", "RIGHT", "EAT", "WAIT"],
    IntelligenceLevel.MONKEY: ["UP", "DOWN", "LEFT", "RIGHT", "EAT", "WAIT"],
    IntelligenceLevel.HUMAN: ["UP", "DOWN", "LEFT", "RIGHT", "EAT", "WAIT"],
}
