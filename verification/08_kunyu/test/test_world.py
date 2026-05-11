import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "impl"))

import pytest
import numpy as np

from entities import EntityType, Action
from event_bus import EventBus
from world import KunyuWorld, WorldConfig


def _make_world(**kwargs):
    w = KunyuWorld(WorldConfig(**kwargs))
    w.register_agent("test_agent")
    return w


def test_world_creation():
    w = _make_world(height=10, width=10)
    assert w.terrain.shape == (10, 10)
    assert w.step_count == 0


def test_agent_spawns_on_empty():
    w = _make_world(height=10, width=10, wall_density=0)
    r, c = w.get_agent_pos()
    assert w.terrain[r, c] == EntityType.EMPTY


def test_move_up():
    w = _make_world(height=10, width=10, wall_density=0, n_food=0, n_dangers=0)
    old_r, old_c = w.get_agent_pos()
    if old_r == 0:
        pytest.skip("Agent at top edge")
    w.step(Action.UP)
    new_r, new_c = w.get_agent_pos()
    assert new_r == old_r - 1
    assert new_c == old_c


def test_wall_collision():
    w = _make_world(height=10, width=10, wall_density=0, n_food=0, n_dangers=0)
    r, c = w.get_agent_pos()
    if r == 0:
        pytest.skip("Agent at top edge")
    w.terrain[r - 1, c] = EntityType.WALL
    old_pos = w.get_agent_pos()
    _, info = w.step(Action.UP)
    assert info["collision"]
    assert w.get_agent_pos() == old_pos


def test_eat_food():
    w = _make_world(height=10, width=10, wall_density=0, n_food=0, n_dangers=0)
    r, c = w.get_agent_pos()
    w.terrain[r, c] = EntityType.FOOD
    _, info = w.step(Action.EAT)
    assert info["energy_gained"] == w.cfg.energy_per_food
    assert w.terrain[r, c] == EntityType.EMPTY


def test_danger_damage():
    w = _make_world(height=10, width=10, wall_density=0, n_food=0, n_dangers=0)
    r, c = w.get_agent_pos()
    w.terrain[r, c] = EntityType.DANGER
    _, info = w.step(Action.WAIT)
    assert info["damage_taken"] == w.cfg.damage_per_danger


def test_move_into_danger():
    w = _make_world(height=10, width=10, wall_density=0, n_food=0, n_dangers=0)
    r, c = w.get_agent_pos()
    if r == 0:
        pytest.skip("Agent at top edge")
    w.terrain[r - 1, c] = EntityType.DANGER
    _, info = w.step(Action.UP)
    assert info["damage_taken"] == w.cfg.damage_per_danger
    assert w.get_agent_pos() == (r - 1, c)


def test_observation_shape():
    w = _make_world(height=10, width=10, vision_range=2)
    obs = w.get_obs()
    assert obs.shape == (5, 5)


def test_agent_in_obs_center():
    w = _make_world(height=10, width=10, vision_range=2)
    obs = w.get_obs()
    R = w.cfg.vision_range
    assert obs[R, R] == EntityType.AGENT


def test_food_visible_in_obs():
    w = _make_world(height=10, width=10, vision_range=2, n_food=1, n_dangers=0, wall_density=0)
    _, info = w.step(Action.WAIT)
    assert info["food_visible"] in (True, False)


def test_get_state():
    w = _make_world()
    state = w.get_state()
    assert "step" in state
    assert "agent_pos" in state
    assert "terrain" in state
    assert isinstance(state["terrain"], np.ndarray)


def test_food_respawn():
    cfg = WorldConfig(height=5, width=5, wall_density=0, n_food=0, n_dangers=0,
                      food_respawn_interval=2, max_food=5)
    w = KunyuWorld(cfg)
    w.register_agent("test_agent")
    n_before = int(np.sum(w.terrain == EntityType.FOOD))
    w.step(Action.WAIT)
    w.step(Action.WAIT)
    n_after = int(np.sum(w.terrain == EntityType.FOOD))
    assert n_after > n_before


def test_multi_agent():
    w = KunyuWorld(WorldConfig(height=10, width=10, wall_density=0, n_food=0, n_dangers=0))
    w.register_agent("alice")
    w.register_agent("bob")
    assert len(w.agent_ids) == 2
    assert w.get_agent_pos("alice") != w.get_agent_pos("bob")
    w.step(Action.UP, agent_id="alice")
    w.step(Action.DOWN, agent_id="bob")
    assert w.get_agent_pos("alice") != w.get_agent_pos("bob")


def test_remove_agent():
    w = KunyuWorld(WorldConfig(height=10, width=10))
    w.register_agent("alice")
    w.register_agent("bob")
    w.remove_agent("alice")
    assert w.agent_ids == ["bob"]


def test_agent_enter_event():
    bus = EventBus()
    events = []
    bus.subscribe("world.agent_enter", lambda d: events.append(d))
    w = KunyuWorld(WorldConfig(height=10, width=10), bus=bus)
    w.register_agent("test")
    assert len(events) == 1
    assert events[0]["agent_id"] == "test"
