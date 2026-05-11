"""Perception — translates raw 5×5 grid into structured Facts.

EntityType values (from 08_kunyu):
  0=EMPTY  1=WALL  2=FOOD  3=DANGER  4=AGENT
"""

import numpy as np

# Local constants to avoid cross-module import issues
_EMPTY = 0
_WALL = 1
_FOOD = 2
_DANGER = 3
_AGENT = 4

_DIRECTIONS = [(-1, 0, "UP"), (1, 0, "DOWN"), (0, -1, "LEFT"), (0, 1, "RIGHT")]


class Facts:
    def __init__(self):
        self.food_dirs: list[str] = []
        self.danger_dirs: list[str] = []
        self.is_new_tile: bool = False
        self.n_walls_near: int = 0
        self.n_open_cells: int = 0


def perceive(obs: np.ndarray, visited: set, agent_pos: tuple) -> Facts:
    facts = Facts()

    R = obs.shape[0] // 2
    center = (R, R)

    for dr, dc, dname in _DIRECTIONS:
        cell_type = obs[center[0] + dr, center[1] + dc]
        if cell_type == _FOOD:
            facts.food_dirs.append(dname)
        elif cell_type == _DANGER:
            facts.danger_dirs.append(dname)

    if obs[center] == _FOOD:
        facts.food_dirs.insert(0, "HERE")
    if obs[center] == _DANGER:
        facts.danger_dirs.insert(0, "HERE")

    for r in range(obs.shape[0]):
        for c in range(obs.shape[1]):
            cell = obs[r, c]
            if cell == _WALL:
                facts.n_walls_near += 1
            elif cell == _AGENT:
                pass  # self
            else:
                facts.n_open_cells += 1

    # Agent pos is relative; check visited via the caller's set
    # (visited is the agent's own memory of where it's been)

    return facts


def compute_novelty(pos: tuple, visited: set, total_cells: int) -> float:
    if total_cells == 0:
        return 0.0
    return max(0.0, 1.0 - len(visited) / total_cells)
