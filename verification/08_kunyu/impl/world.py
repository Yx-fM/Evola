import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from entities import EntityType, Action, ACTION_DELTA, ACTION_NAME
from event_bus import EventBus


@dataclass
class WorldConfig:
    height: int = 20
    width: int = 20
    vision_range: int = 2
    n_food: int = 5
    n_dangers: int = 3
    food_respawn_interval: int = 20
    max_food: int = 8
    wall_density: float = 0.05
    energy_per_food: float = 0.3
    damage_per_danger: float = 0.2


@dataclass
class _AgentRecord:
    pos: tuple[int, int] = (0, 0)
    visited: set = field(default_factory=set)


class KunyuWorld:
    def __init__(self, config: Optional[WorldConfig] = None, bus: Optional[EventBus] = None):
        self.cfg = config or WorldConfig()
        self.bus = bus or EventBus()
        self.terrain = np.zeros((self.cfg.height, self.cfg.width), dtype=np.int32)
        self._agents: dict[str, _AgentRecord] = {}
        self._default_agent: Optional[str] = None
        self.step_count = 0
        self.food_spawn_timer = 0
        self._init_world()

    def _init_world(self):
        rng = np.random.default_rng()
        h, w = self.cfg.height, self.cfg.width

        self.terrain.fill(EntityType.EMPTY)

        n_wall_cells = int(h * w * self.cfg.wall_density)
        wall_indices = rng.choice(h * w, size=n_wall_cells, replace=False)
        self.terrain.flat[wall_indices] = EntityType.WALL

        empty_mask = self.terrain == EntityType.EMPTY
        empty_indices = np.flatnonzero(empty_mask.ravel())

        food_idx = rng.choice(empty_indices, size=self.cfg.n_food, replace=False)
        self.terrain.flat[food_idx] = EntityType.FOOD

        remaining = np.setdiff1d(empty_indices, food_idx)
        danger_idx = rng.choice(remaining, size=self.cfg.n_dangers, replace=False)
        self.terrain.flat[danger_idx] = EntityType.DANGER

        self.bus.publish("world.init", {
            "height": h, "width": w,
            "n_food": self.cfg.n_food, "n_dangers": self.cfg.n_dangers,
        })

    def register_agent(self, agent_id: str, pos: Optional[tuple[int, int]] = None):
        if pos is None:
            pos = self._find_empty_cell()
        rec = _AgentRecord(pos=pos)
        rec.visited.add(pos)
        self._agents[agent_id] = rec
        if self._default_agent is None:
            self._default_agent = agent_id
        self.bus.publish("world.agent_enter", {"agent_id": agent_id, "pos": pos})

    def remove_agent(self, agent_id: str):
        if agent_id in self._agents:
            del self._agents[agent_id]
        if self._default_agent == agent_id:
            self._default_agent = next(iter(self._agents), None)
        self.bus.publish("world.agent_leave", {"agent_id": agent_id})

    @property
    def agent_ids(self):
        return list(self._agents.keys())

    def get_agent_pos(self, agent_id: Optional[str] = None):
        aid = agent_id or self._default_agent
        if aid is None:
            return (0, 0)
        return self._agents[aid].pos

    def step(self, action: Action, dt: float = 1.0, agent_id: Optional[str] = None):
        aid = agent_id or self._default_agent
        if aid is None:
            raise RuntimeError("No agent registered. Call register_agent() first.")

        self.step_count += 1
        rec = self._agents[aid]
        dr, dc = ACTION_DELTA.get(action, (0, 0))
        r, c = rec.pos
        nr, nc = r + dr, c + dc
        collision = False
        food_eaten = False
        damage_taken = False

        self.bus.publish("step.begin", {
            "step": self.step_count, "agent_id": aid,
            "agent_pos": rec.pos, "action": ACTION_NAME.get(action, "?"),
        })

        if action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT):
            if self._in_bounds(nr, nc) and self.terrain[nr, nc] != EntityType.WALL:
                rec.pos = (nr, nc)
                rec.visited.add((nr, nc))
            else:
                collision = True

        r, c = rec.pos
        terrain_here = self.terrain[r, c]

        if terrain_here == EntityType.DANGER:
            damage_taken = True

        if action == Action.EAT and terrain_here == EntityType.FOOD:
            food_eaten = True
            self.terrain[r, c] = EntityType.EMPTY

        self._respawn_food()
        obs = self.get_obs(aid)
        info = {
            "agent_id": aid,
            "energy_gained": float(food_eaten) * self.cfg.energy_per_food,
            "damage_taken": float(damage_taken) * self.cfg.damage_per_danger,
            "collision": collision,
            "food_visible": EntityType.FOOD in obs,
            "danger_visible": EntityType.DANGER in obs,
        }

        self.bus.publish("step.outcome", {
            "step": self.step_count, "agent_id": aid,
            "agent_pos": rec.pos,
            "energy_gained": info["energy_gained"],
            "damage_taken": info["damage_taken"],
            "collision": collision,
            "food_visible": info["food_visible"],
            "danger_visible": info["danger_visible"],
        })
        self.bus.publish("step.end", {
            "step": self.step_count, "agent_id": aid, "agent_pos": rec.pos,
        })

        return obs, info

    def get_obs(self, agent_id: Optional[str] = None):
        aid = agent_id or self._default_agent
        if aid is None:
            return np.zeros((0, 0), dtype=np.int32)
        rec = self._agents[aid]
        R = self.cfg.vision_range
        obs = np.full((2 * R + 1, 2 * R + 1), EntityType.EMPTY, dtype=np.int32)
        r_center, c_center = rec.pos
        for dr in range(-R, R + 1):
            for dc in range(-R, R + 1):
                wr, wc = r_center + dr, c_center + dc
                if self._in_bounds(wr, wc):
                    if (wr, wc) == rec.pos:
                        obs[dr + R, dc + R] = EntityType.AGENT
                    else:
                        obs[dr + R, dc + R] = self.terrain[wr, wc]
        return obs

    def get_full_grid(self):
        grid = self.terrain.copy()
        for aid, rec in self._agents.items():
            r, c = rec.pos
            grid[r, c] = EntityType.AGENT
        return grid

    def get_state(self, agent_id: Optional[str] = None):
        aid = agent_id or self._default_agent
        rec = self._agents.get(aid) if aid else None
        return {
            "step": self.step_count,
            "agent_id": aid,
            "agent_pos": rec.pos if rec else None,
            "terrain": self.terrain.copy(),
            "grid": self.get_full_grid(),
            "n_agents": len(self._agents),
            "n_visited": len(rec.visited) if rec else 0,
            "config": self.cfg,
        }

    def get_visited(self, agent_id: Optional[str] = None):
        aid = agent_id or self._default_agent
        rec = self._agents.get(aid)
        return rec.visited.copy() if rec else set()

    def _find_empty_cell(self):
        empty_cells = np.flatnonzero(
            (self.terrain.ravel() == EntityType.EMPTY)
        )
        for aid, rec in self._agents.items():
            idx = rec.pos[0] * self.cfg.width + rec.pos[1]
            empty_cells = empty_cells[empty_cells != idx]
        if len(empty_cells) == 0:
            raise RuntimeError("No empty cells available")
        idx = np.random.default_rng().choice(empty_cells)
        return (idx // self.cfg.width, idx % self.cfg.width)

    def _respawn_food(self):
        self.food_spawn_timer += 1
        if self.food_spawn_timer < self.cfg.food_respawn_interval:
            return
        self.food_spawn_timer = 0
        n_current = int(np.sum(self.terrain == EntityType.FOOD))
        if n_current >= self.cfg.max_food:
            return
        occupied = {rec.pos for rec in self._agents.values()}
        empty_cells = np.flatnonzero(self.terrain.ravel() == EntityType.EMPTY)
        empty_cells = [i for i in empty_cells
                       if (i // self.cfg.width, i % self.cfg.width) not in occupied]
        if len(empty_cells) == 0:
            return
        idx = np.random.default_rng().choice(empty_cells)
        self.terrain.flat[idx] = EntityType.FOOD

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.cfg.height and 0 <= c < self.cfg.width
