"""SensorimotorAgent — IAgent using 10_sensorimotor + 02_homeostasis.

Perceives the world via 10, decides via behavior standards,
and feels needs via 02 (homeostatic drive system).

Supports two arbiter types:
  "rule" (P0) — hardcoded RuleArbiter
  "ltc"  (P1) — LTCArbiter with online learning
"""

import sys, os

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_BASE, "..", "08_kunyu", "impl"))
sys.path.insert(0, os.path.join(_BASE, "..", "02_homeostasis", "impl"))

sys.path.insert(0, os.path.join(_BASE, "..", "05_world_model", "impl"))

import numpy as np
import torch
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
                 safety_rise: float = 0.8,
                 safety_decay: float = 0.05,
                 arbiter_type: str = "rule",
                 arbiter_params: dict | None = None,
                 ltc_hidden: int = 64,
                 ltc_lr: float = 0.001,
                 rng=None):
        self._id = agent_id
        self.step_count = 0
        self.last_info = {}
        self.visited: set = set()
        self._last_pos = (0, 0)
        self._arbiter_type = arbiter_type
        self._rng = rng or np.random.default_rng()
        self._last_obs = None  # for world model training
        self._last_action = None
        self._wm = None  # lazily created

        # Homeostasis (02) — her body / needs
        self.homeo = Homeostasis(energy_decay=energy_decay,
                                  safety_rise=safety_rise,
                                  safety_decay=safety_decay)

        # Arbiter — her decision system (P0: rule, P1: LTC)
        if arbiter_type == "ltc":
            sys.path.insert(0, os.path.join(_BASE, "..", "01_liquid_dynamics", "impl"))
            from ltc_arbiter import LTCArbiter
            self.arbiter = LTCArbiter(hidden_size=ltc_hidden, lr=ltc_lr)
        else:
            ap = arbiter_params or {}
            self.arbiter = RuleArbiter(
                safety_threshold=ap.get("safety_threshold", 0.3),
                energy_urgent=ap.get("energy_urgent", 0.5),
                energy_safe=ap.get("energy_safe", 0.3),
                novelty_threshold=ap.get("novelty_threshold", 0.2),
            )

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
        if self._arbiter_type == "ltc" and hasattr(self.arbiter, 'set_context_pos'):
            self.arbiter.set_context_pos(self._last_pos)
        primitive = self.arbiter.select(drive, arb_facts, SalienceMap(), context)

        # 5. Actuator: primitive → raw action
        action = self.actuator.execute(primitive, arb_facts)
        self._last_obs = obs
        self._last_action = int(action)
        return action

    def observe(self, info: dict) -> None:
        self.last_info = info

        agent_pos = info.get("agent_pos")
        new_tile = False
        if agent_pos:
            pt = tuple(agent_pos)
            self._last_pos = pt
            if pt not in self.visited:
                new_tile = True
            self.visited.add(pt)

        food_eaten = info.get("energy_gained", 0) > 0
        damage_taken = info.get("damage_taken", 0) > 0

        info["_novelty_visited_count"] = len(self.visited)
        self.homeo.update(info)

        # World model training
        if self._last_obs is not None and self._last_action is not None:
            current_obs = np.asarray(self._last_obs, dtype=np.float32).ravel()
            # Use info to estimate next_obs heuristically
            next_obs = current_obs.copy()
            if food_eaten:
                next_obs[0] = 0.0  # food at HERE gone
            self._train_wm(torch.tensor(current_obs), self._last_action,
                           torch.tensor(next_obs))
        self._last_obs = None
        self._last_action = None

        if self._arbiter_type == "ltc":
            h = self.homeo
            self.arbiter.update_self(h.energy.value, h.novelty.value, h.safety.value,
                                      food_eaten, damage_taken, new_tile)
            self.arbiter.update_memory(self.step_count, pt if pt else (0, 0),
                                        h.energy.value, h.novelty.value, h.safety.value,
                                        food_eaten, damage_taken, new_tile,
                                        info.get("food_visible", False))
            drive = self.homeo.get_drive()
            self.arbiter.update(drive)

    def _init_wm(self):
        if self._wm is None:
            from world_model import WorldModel
            self._wm = WorldModel()
            self._wm_opt = torch.optim.Adam(self._wm.parameters(), lr=0.001)

    def _train_wm(self, obs_t, action, next_obs_t):
        if self._wm is None:
            self._init_wm()
        self._wm.train_step(obs_t, action, next_obs_t, self._wm_opt)

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

    def save_mind(self, path: str):
        """Persist complete agent state to .mind.evola."""
        import json, base64, io

        def _clean(obj):
            if isinstance(obj, dict): return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list): return [_clean(v) for v in obj]
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            return obj
        data = {
            "format": "evola.mind", "version": "mvp2.0",
            "agent_id": self._id, "step_count": self.step_count,
            "arbiter_type": self._arbiter_type,
        }

        # Homeostasis
        h = self.homeo
        data["homeostasis"] = {
            "energy": h.energy.value, "safety": h.safety.value,
            "novelty_visited_count": len(self.visited),
            "novelty_total_cells": h.novelty.total_cells,
        }

        # Visited
        data["visited"] = [[int(p[0]), int(p[1])] for p in self.visited]

        # LTC weights (base64)
        if self._arbiter_type == "ltc":
            buf = io.BytesIO()
            torch.save(self.arbiter.net.state_dict(), buf)
            data["ltc_weights"] = base64.b64encode(buf.getvalue()).decode("ascii")

        # Memory items
        if self._arbiter_type == "ltc" and getattr(self.arbiter, 'memory', None):
            m = self.arbiter.memory
            data["memory"] = {
                "stm": [{"step": i.step, "pos": list(i.pos), "event_type": i.event_type,
                          "importance": i.importance, "decay": i.decay,
                          "energy": i.energy, "novelty": i.novelty, "safety": i.safety}
                         for i in m.stm],
                "ltm": [{"step": i.step, "pos": list(i.pos), "event_type": i.event_type,
                          "importance": i.importance, "decay": i.decay,
                          "energy": i.energy, "novelty": i.novelty, "safety": i.safety}
                         for i in m.ltm],
            }

        # Self model
        if self._arbiter_type == "ltc" and self.arbiter.self_model:
            sm = self.arbiter.self_model
            data["self_model"] = {
                "slow": sm.slow.tolist(), "fast": sm.fast.tolist(),
            }

        # World model
        if self._wm:
            buf = io.BytesIO()
            torch.save(self._wm.state_dict(), buf)
            data["wm_weights"] = base64.b64encode(buf.getvalue()).decode("ascii")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(_clean(data), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_mind(cls, path: str, **kwargs):
        """Restore agent from .mind.evola."""
        import json, base64, io
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Use stored arbiter_type, allow override
        arb_type = kwargs.pop("arbiter_type", data.get("arbiter_type", "rule"))
        arb_type = data.get("arbiter_type", arb_type)

        agent = cls(
            agent_id=data["agent_id"],
            arbiter_type=arb_type,
            **kwargs,
        )
        agent.step_count = data.get("step_count", 0)

        # Homeostasis
        hs = data.get("homeostasis", {})
        if hs:
            h = agent.homeo
            h.energy.value = hs.get("energy", 0.8)
            h.safety.value = hs.get("safety", 0.0)
            h.novelty.total_cells = hs.get("novelty_total_cells", 300)
            h.novelty.set_visited(hs.get("novelty_visited_count", 0))

        # Visited
        for p in data.get("visited", []):
            agent.visited.add((p[0], p[1]))

        # LTC weights
        if "ltc_weights" in data and agent._arbiter_type == "ltc":
            buf = io.BytesIO(base64.b64decode(data["ltc_weights"]))
            agent.arbiter.net.load_state_dict(torch.load(buf))

        # Memory
        if "memory" in data and agent._arbiter_type == "ltc" and agent.arbiter.memory:
            sys.path.insert(0, os.path.join(_BASE, "..", "11_memory_io", "impl"))
            from memory_store import MemoryItem
            m = agent.arbiter.memory
            m.stm.clear(); m.ltm.clear()
            for item_data in data["memory"].get("stm", []):
                item = MemoryItem(
                    id=0,
                    step=int(item_data["step"]),
                    pos=tuple(item_data["pos"]),
                    event_type=item_data["event_type"],
                    energy=float(item_data.get("energy", 0.5)),
                    novelty=float(item_data.get("novelty", 0.5)),
                    safety=float(item_data.get("safety", 0.0)),
                    importance=float(item_data["importance"]),
                    decay=float(item_data.get("decay", 1.0)),
                )
                m.stm.append(item)
            for item_data in data["memory"].get("ltm", []):
                item = MemoryItem(
                    id=0,
                    step=int(item_data["step"]),
                    pos=tuple(item_data["pos"]),
                    event_type=item_data["event_type"],
                    energy=float(item_data.get("energy", 0.5)),
                    novelty=float(item_data.get("novelty", 0.5)),
                    safety=float(item_data.get("safety", 0.0)),
                    importance=float(item_data["importance"]),
                    decay=float(item_data.get("decay", 1.0)),
                )
                m.ltm.append(item)

        # Self model
        if "self_model" in data and agent._arbiter_type == "ltc" and agent.arbiter.self_model:
            sm = agent.arbiter.self_model
            sm.slow = np.array(data["self_model"]["slow"], dtype=np.float32)
            sm.fast = np.array(data["self_model"]["fast"], dtype=np.float32)
            sm._initialized = True

        # World model
        if "wm_weights" in data:
            agent._init_wm()
            buf = io.BytesIO(base64.b64decode(data["wm_weights"]))
            agent._wm.load_state_dict(torch.load(buf))

        return agent
