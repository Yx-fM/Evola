"""SensorimotorAgent — IAgent using 10_sensorimotor + 02_homeostasis.

Perceives the world via 10, decides via behavior standards,
and feels needs via 02 (homeostatic drive system).

Architecture:
  08_kunyu → obs → [10_sensorimotor → facts] → [02_homeostasis → drive] → [10_arbiter → action]
"""

import sys, os

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_BASE, "..", "08_kunyu", "impl"))
sys.path.insert(0, os.path.join(_BASE, "..", "02_homeostasis", "impl"))

import numpy as np
from agent_interface import IAgent
from behavior_standard.primitives import BehaviorPrimitive
from behavior_standard.arbitration import (
    RuleArbiter, Facts as ArbFacts, SalienceMap, DecisionContext,
)
from behavior_standard.actuator import Actuator
from perception.perception import perceive
from perception.salience import compute_salience
from drives import Homeostasis, DriveVector


class SensorimotorAgent(IAgent):
    def __init__(self, agent_id: str = "evo",
                 energy_decay: float = 0.003,
                 rng=None):
        self._id = agent_id
        self.step_count = 0
        self.last_info = {}
        self.visited: set = set()

        # Homeostasis (02) — her body / needs
        self.homeo = Homeostasis(energy_decay=energy_decay)
        self._rng = rng or np.random.default_rng()

        # Behavior standard (10) — her decision system
        self.arbiter = RuleArbiter()
        self.actuator = Actuator(self._rng)

    @property
    def agent_id(self) -> str:
        return self._id

    def act(self, obs: np.ndarray) -> int:
        self.step_count += 1

        # 1. Perceive
        facts = perceive(obs, self.visited, None)

        # 2. Get drive from homeostasis
        drive = self.homeo.get_drive()

        # 3. Salience
        salience = compute_salience(drive.energy, drive.safety, drive.novelty)

        # 4. Arbiter: drive + facts → primitive
        arb_facts = ArbFacts(
            food_dirs=facts.food_dirs,
            danger_dirs=facts.danger_dirs,
            is_new_tile=facts.is_new_tile,
            n_walls_near=facts.n_walls_near,
            n_open_cells=facts.n_open_cells,
        )
        context = DecisionContext()
        primitive = self.arbiter.select(drive, arb_facts, SalienceMap(), context)

        # 5. Actuator: primitive → raw action
        action = self.actuator.execute(primitive, arb_facts)
        return action

    def observe(self, info: dict) -> None:
        self.last_info = info

        # Feed world outcome into homeostasis
        self.homeo.update(info)

        # Track visited cells
        agent_pos = info.get("agent_pos")
        if agent_pos:
            self.visited.add(tuple(agent_pos))

    def set_total_cells(self, n: int):
        self.homeo.novelty.total_cells = n

    def get_state(self) -> dict:
        h = self.homeo.get_state()
        return {
            "agent_id": self._id,
            "step_count": self.step_count,
            "homeostasis": h,
            "n_visited": len(self.visited),
        }
