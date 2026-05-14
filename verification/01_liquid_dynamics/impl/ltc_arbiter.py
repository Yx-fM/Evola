"""LTCArbiter — IArbiter with LTC network + SelfModel + Memory.

P0→P1→P2 progression:
  facts(25) + drive(3) + self_vector(8) + memory_ctx(4) = 40 → LTC → action
"""

import sys, os
_IMPL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_IMPL, "..", "..", "10_sensorimotor", "impl"))
sys.path.insert(0, os.path.join(_IMPL, "..", "..", "04_self_model", "impl"))
sys.path.insert(0, os.path.join(_IMPL, "..", "..", "11_memory_io", "impl"))

import numpy as np
import torch
import torch.nn.functional as F
from behavior_standard.arbitration import IArbiter, DriveVector, Facts, SalienceMap, DecisionContext
from behavior_standard.primitives import BehaviorPrimitive
from ltc_network import LTCNetwork
from self_model import SelfModel
from memory_io import MemoryIO
from memory_store import ActiveMemory


class LTCArbiter(IArbiter):
    def __init__(self, hidden_size: int = 64, lr: float = 0.001,
                 use_self_model: bool = True, use_memory: bool = True,
                 device: str = "cpu"):
        self.hidden_size = hidden_size
        self.device = device
        self.use_self_model = use_self_model
        self.use_memory = use_memory

        dims = 25 + 3
        if use_self_model: dims += 8
        if use_memory: dims += 4

        self.net = LTCNetwork(input_size=dims, hidden_size=hidden_size).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.self_model = SelfModel() if use_self_model else None
        self.memory = ActiveMemory() if use_memory else None
        self.mem_io = MemoryIO(self.memory) if use_memory else None

        self.h = None
        self._prev_drive_sum = 0.0
        self._log_prob = None
        self._action_idx = 0
        self._context_pos = None

    def set_context_pos(self, pos):
        self._context_pos = pos

    def _facts_to_obs(self, facts: Facts) -> np.ndarray:
        obs = np.zeros(25, dtype=np.float32)
        for d in facts.food_dirs:
            obs[{"UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4, "HERE": 0}.get(d, 0)] = 1.0
        for d in facts.danger_dirs:
            obs[{"UP": 6, "DOWN": 7, "LEFT": 8, "RIGHT": 9, "HERE": 5}.get(d, 0)] = 1.0
        obs[10] = 1.0 if facts.is_new_tile else 0.0
        obs[11] = float(facts.n_walls_near) / 25.0
        obs[12] = float(facts.n_open_cells) / 25.0
        return obs

    def select(self, drive: DriveVector, facts: Facts,
               salience: SalienceMap, context: DecisionContext) -> BehaviorPrimitive:
        obs = self._facts_to_obs(facts)
        d = np.array([drive.energy, drive.novelty, drive.safety], dtype=np.float32)

        obs_t = torch.tensor(obs, device=self.device).unsqueeze(0)
        drive_t = torch.tensor(d, device=self.device).unsqueeze(0)

        self_vec_t = None
        mem_t = None

        if self.self_model is not None:
            sv = self.self_model.encode()
            self_vec_t = torch.tensor(sv, device=self.device).unsqueeze(0)

        if self.memory is not None:
            mem_ctx = self.mem_io.retrieve_context(self._context_pos, k=4)
            mem_t = torch.tensor(mem_ctx, device=self.device).unsqueeze(0)

        if self.h is None:
            self.h = self.net.reset_hidden(1).to(self.device)
        h_detached = self.h.detach()

        logits, _, self.h = self.net(obs_t, drive_t, h_detached,
                                      self_vec=self_vec_t, mem=mem_t)
        self._prev_drive_sum = drive.energy + drive.novelty + drive.safety
        self._log_prob = F.log_softmax(logits, dim=-1).squeeze(0)
        probs = F.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
        self._action_idx = np.random.choice(6, p=probs)
        self.h = self.h.detach()
        return self._action_to_primitive(self._action_idx)

    def update(self, drive: DriveVector):
        drive_sum = drive.energy + drive.novelty + drive.safety
        reward = self._prev_drive_sum - drive_sum
        if self._log_prob is None: return
        loss = -self._log_prob[self._action_idx] * reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_self(self, energy: float, novelty: float, safety: float,
                    food_eaten: bool, damage_taken: bool, new_tile: bool):
        if self.self_model is not None:
            self.self_model.update(energy, novelty, safety, food_eaten, damage_taken, new_tile)

    def update_memory(self, step: int, pos: tuple[int, int],
                      energy: float, novelty: float, safety: float,
                      food_eaten: bool, damage_taken: bool, new_tile: bool,
                      food_visible: bool):
        if self.mem_io is not None:
            self.mem_io.encode(step, pos, energy, novelty, safety,
                               food_eaten, damage_taken, new_tile, food_visible)
            load_drive = self.memory.get_memory_load()
            self.memory.decay_step(load_drive)

    def reset_hidden(self):
        self.h = None

    def _action_to_primitive(self, idx: int) -> BehaviorPrimitive:
        if idx == 4: return BehaviorPrimitive.CONSUME
        if idx == 5: return BehaviorPrimitive.WAIT
        return BehaviorPrimitive.WANDER

    def state_dict(self) -> dict:
        return {"net": self.net.state_dict()}

    def load_state_dict(self, d: dict):
        self.net.load_state_dict(d["net"])

    def train(self): self.net.train()
    def eval(self): self.net.eval()
