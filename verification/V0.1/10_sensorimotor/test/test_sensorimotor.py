import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "impl"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "08_kunyu", "impl"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "02_homeostasis", "impl"))

import numpy as np
import pytest

from perception.perception import perceive, compute_novelty
from perception.salience import compute_salience
from behavior_standard.primitives import BehaviorPrimitive
from behavior_standard.arbitration import (
    RuleArbiter, DriveVector, Facts, SalienceMap, DecisionContext,
)
from behavior_standard.actuator import Actuator
from agent import SensorimotorAgent


def _make_obs(food_dirs=None, danger_dirs=None):
    """Build a 5×5 observation. AGENT always at (2,2)."""
    EMPTY, WALL, FOOD, DANGER, AGENT = 0, 1, 2, 3, 4
    obs = np.zeros((5, 5), dtype=np.int32)
    obs[2, 2] = AGENT
    dir_map = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 1), "RIGHT": (1, 1)}
    for d in (food_dirs or []):
        r, c = dir_map.get(d, (0, 0))
        obs[2 + r, 2 + c] = FOOD
    for d in (danger_dirs or []):
        r, c = dir_map.get(d, (0, 0))
        obs[2 + r, 2 + c] = DANGER
    return obs


class TestPerception:
    def test_empty_obs(self):
        obs = np.full((5, 5), 0, dtype=np.int32)
        obs[2, 2] = 4  # AGENT
        facts = perceive(obs, set(), (2, 2))
        assert facts.food_dirs == []
        assert facts.danger_dirs == []
        assert facts.n_walls_near == 0
        assert facts.n_open_cells == 24

    def test_food_up(self):
        EMPTY, WALL, FOOD, DANGER, AGENT = 0, 1, 2, 3, 4
        obs = np.zeros((5, 5), dtype=np.int32)
        obs[2, 2] = AGENT
        obs[1, 2] = FOOD
        facts = perceive(obs, set(), (2, 2))
        assert "UP" in facts.food_dirs

    def test_danger_left(self):
        EMPTY, WALL, FOOD, DANGER, AGENT = 0, 1, 2, 3, 4
        obs = np.zeros((5, 5), dtype=np.int32)
        obs[2, 2] = AGENT
        obs[2, 1] = DANGER
        facts = perceive(obs, set(), (2, 2))
        assert "LEFT" in facts.danger_dirs

    def test_food_here(self):
        EMPTY, WALL, FOOD, DANGER, AGENT = 0, 1, 2, 3, 4
        obs = np.zeros((5, 5), dtype=np.int32)
        obs[2, 2] = FOOD
        facts = perceive(obs, set(), (2, 2))
        assert "HERE" in facts.food_dirs

    def test_wall_count(self):
        EMPTY, WALL, FOOD, DANGER, AGENT = 0, 1, 2, 3, 4
        obs = np.ones((5, 5), dtype=np.int32)  # all walls
        obs[2, 2] = AGENT
        facts = perceive(obs, set(), (2, 2))
        assert facts.n_walls_near == 24
        assert facts.n_open_cells == 0


class TestSalience:
    def test_hungry_makes_food_relevant(self):
        sm = compute_salience(0.9, 0.0, 0.1)
        assert sm.food_relevance > 0.8
        assert sm.danger_relevance < 0.1

    def test_danger_makes_danger_relevant(self):
        sm = compute_salience(0.0, 0.8, 0.0)
        assert sm.danger_relevance > 0.7
        assert sm.food_relevance < 0.1


class TestArbiter:
    def test_hungry_chooses_approach(self):
        arbiter = RuleArbiter()
        drive = DriveVector(energy=0.9, novelty=0.1, safety=0.0)
        facts = Facts(food_dirs=["UP"])
        primitive = arbiter.select(drive, facts, SalienceMap(), DecisionContext())
        assert primitive == BehaviorPrimitive.APPROACH

    def test_food_here_chooses_consume(self):
        arbiter = RuleArbiter()
        drive = DriveVector(energy=0.9, novelty=0.1, safety=0.0)
        facts = Facts(food_dirs=["HERE", "UP"])
        primitive = arbiter.select(drive, facts, SalienceMap(), DecisionContext())
        assert primitive == BehaviorPrimitive.CONSUME

    def test_danger_chooses_avoid(self):
        arbiter = RuleArbiter()
        drive = DriveVector(energy=0.3, novelty=0.1, safety=0.8)
        facts = Facts(danger_dirs=["LEFT"], food_dirs=["UP"])
        primitive = arbiter.select(drive, facts, SalienceMap(), DecisionContext())
        assert primitive == BehaviorPrimitive.AVOID

    def test_not_hungry_explores(self):
        arbiter = RuleArbiter()
        drive = DriveVector(energy=0.1, novelty=0.8, safety=0.0)
        facts = Facts(food_dirs=["UP"])
        primitive = arbiter.select(drive, facts, SalienceMap(), DecisionContext())
        assert primitive == BehaviorPrimitive.EXPLORE

    def test_default_wanders(self):
        arbiter = RuleArbiter()
        drive = DriveVector(energy=0.1, novelty=0.1, safety=0.0)
        facts = Facts()
        primitive = arbiter.select(drive, facts, SalienceMap(), DecisionContext())
        assert primitive == BehaviorPrimitive.WANDER


class TestActuator:
    def test_consume_returns_eat(self):
        actuator = Actuator()
        action = actuator.execute(BehaviorPrimitive.CONSUME, Facts())
        assert action == 4

    def test_wait_returns_wait(self):
        actuator = Actuator()
        action = actuator.execute(BehaviorPrimitive.WAIT, Facts())
        assert action == 5

    def test_approach_food_moves_toward_food(self):
        actuator = Actuator()
        facts = Facts(food_dirs=["RIGHT"])
        action = actuator.execute(BehaviorPrimitive.APPROACH, facts)
        assert action == 3  # RIGHT

    def test_avoid_danger_moves_opposite(self):
        actuator = Actuator()
        facts = Facts(danger_dirs=["UP"])
        action = actuator.execute(BehaviorPrimitive.AVOID, facts)
        assert action == 1  # DOWN (opposite of UP)

    def test_wander_returns_valid_action(self):
        actuator = Actuator()
        for _ in range(20):
            action = actuator.execute(BehaviorPrimitive.WANDER, Facts())
            assert 0 <= action <= 5


class TestAgent:
    def test_sensorimotor_agent_creation(self):
        agent = SensorimotorAgent("test_agent")
        assert agent.agent_id == "test_agent"

    def test_agent_acts_on_empty_obs(self):
        agent = SensorimotorAgent("test")
        obs = np.zeros((5, 5), dtype=np.int32)
        obs[2, 2] = 4  # AGENT
        action = agent.act(obs)
        assert 0 <= action <= 5

    def test_agent_eats_nearby_food(self):
        agent = SensorimotorAgent("test")
        obs = np.zeros((5, 5), dtype=np.int32)
        obs[2, 2] = 4  # AGENT
        obs[1, 2] = 2  # FOOD 1 cell away
        agent.set_total_cells(100)
        agent.homeo.energy.value = 0.3  # hungry
        actions = set()
        for _ in range(10):
            actions.add(agent.act(obs))
        assert 0 in actions or 4 in actions  # UP or EAT

    def test_observe_updates_energy(self):
        agent = SensorimotorAgent("test")
        agent.homeo.energy.value = 0.5
        agent.observe({"energy_gained": 0.3, "agent_pos": (5, 5)})
        assert agent.homeo.energy.value > 0.7

    def test_observe_updates_visited(self):
        agent = SensorimotorAgent("test")
        agent.observe({"agent_pos": (5, 5)})
        assert (5, 5) in agent.visited
