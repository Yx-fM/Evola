"""Agent modes — hot/warm/cold state definitions.

HOT  : full-frequency action, perception, and learning
WARM : no action, but internal integration + memory consolidation  
COLD : suspended, state preserved for serialization
"""

from enum import Enum
from dataclasses import dataclass


class AgentMode(Enum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


@dataclass
class FreqConfig:
    perception: float = 1.0
    ltc_forward: float = 1.0
    ltc_learning: float = 1.0
    memory_consolidation: float = 0.5
    world_model_train: float = 0.5
    self_model_update: float = 1.0


FREQ_TABLE = {
    AgentMode.HOT: FreqConfig(1.0, 1.0, 1.0, 0.5, 0.5, 1.0),
    AgentMode.WARM: FreqConfig(1.0, 0.3, 0.0, 1.0, 1.0, 0.3),
    AgentMode.COLD: FreqConfig(0, 0, 0, 0, 0, 0),
}
