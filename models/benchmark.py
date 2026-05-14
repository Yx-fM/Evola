"""Evola v0.1 Benchmark — 对比测试不同智能体策略。

Tests: Random | Rule(MVP0) | LTC untrained | LTC trained
Measures: survival, energy stability, exploration, food efficiency
"""

import sys, os, json, time
from collections import deque

_PROJ = r"Q:\All_Items\DreamProjects\Evola"

MODULE_PATHS = [
    os.path.join(_PROJ, "verification", "V0.1", "08_kunyu", "impl"),
    os.path.join(_PROJ, "verification", "V0.1", "02_homeostasis", "impl"),
    os.path.join(_PROJ, "verification", "V0.1", "10_sensorimotor", "impl"),
]
for p in MODULE_PATHS:
    sys.path.insert(0, p)

import numpy as np
from event_bus import EventBus
from world import KunyuWorld, WorldConfig
from agent import SensorimotorAgent
from agent_interface import RandomAgent

WORLD_CFG = {"height": 15, "width": 20, "n_food": 8, "n_dangers": 2,
             "food_respawn_interval": 15, "wall_density": 0.03,
             "energy_per_food": 0.4, "damage_per_danger": 0.15}
MAX_STEPS = 2000
SEED = 42


def run_agent(agent, world, steps):
    """Run agent for N steps, return metrics."""
    energy_history = []
    positions = set()
    food_eaten = 0
    died = False
    death_step = steps

    for s in range(steps):
        obs = world.get_obs(agent.agent_id)
        action = int(agent.act(obs))
        _, info = world.step(action, agent_id=agent.agent_id)
        agent.observe(info)
        if hasattr(agent, 'homeo'):
            energy_history.append(agent.homeo.energy.value)
            if info.get("energy_gained", 0) > 0:
                food_eaten += 1
            if agent.homeo.energy.value <= 0:
                died = True
                death_step = s + 1
                break
        positions.add(world.get_agent_pos(agent.agent_id))

    total = world.cfg.height * world.cfg.width
    return {
        "survived": not died,
        "death_step": death_step,
        "energy_mean": float(np.mean(energy_history)) if energy_history else 0,
        "energy_std": float(np.std(energy_history)) if energy_history else 0,
        "exploration": len(positions) / total,
        "food_eaten": food_eaten,
        "efficiency": food_eaten / max(1, death_step) * 1000,
    }


def test_random():
    bus = EventBus()
    world = KunyuWorld(WorldConfig(**WORLD_CFG), bus=bus)
    agent = RandomAgent("random")
    world.register_agent("random")
    return run_agent(agent, world, MAX_STEPS)


def test_rule():
    bus = EventBus()
    world = KunyuWorld(WorldConfig(**WORLD_CFG), bus=bus)
    agent = SensorimotorAgent("rule", arbiter_type="rule", arbiter_params={
        "safety_threshold": 0.3, "energy_urgent": 0.5, "energy_safe": 0.3, "novelty_threshold": 0.2,
    })
    agent.set_total_cells(world.cfg.height * world.cfg.width)
    world.register_agent("rule")
    return run_agent(agent, world, MAX_STEPS)


def test_ltc_untrained():
    bus = EventBus()
    world = KunyuWorld(WorldConfig(**WORLD_CFG), bus=bus)
    agent = SensorimotorAgent("ltc0", arbiter_type="ltc", ltc_hidden=64, ltc_lr=0.001,
                               energy_decay=0.003)
    agent.set_total_cells(world.cfg.height * world.cfg.width)
    world.register_agent("ltc0")
    return run_agent(agent, world, MAX_STEPS)


def test_ltc_trained(pretrain_steps=500):
    """Load untrained model, run pretrain_steps, then test on fresh world."""
    mind_path = os.path.join(_PROJ, "models", "v0.1", "evola_v0.1_untrained.mind.evola")
    agent = SensorimotorAgent.load_mind(mind_path, arbiter_params={
        "safety_threshold": 0.3, "energy_urgent": 0.5, "energy_safe": 0.3, "novelty_threshold": 0.2,
    })

    # Pretrain
    bus = EventBus()
    world = KunyuWorld(WorldConfig(**WORLD_CFG), bus=bus)
    agent.set_total_cells(world.cfg.height * world.cfg.width)
    world.register_agent(agent.agent_id)
    for _ in range(pretrain_steps):
        obs = world.get_obs(agent.agent_id)
        action = int(agent.act(obs))
        _, info = world.step(action, agent_id=agent.agent_id)
        agent.observe(info)

    # Test on fresh world
    bus2 = EventBus()
    world2 = KunyuWorld(WorldConfig(**WORLD_CFG), bus=bus2)
    agent2 = SensorimotorAgent("ltc_trained", arbiter_type="ltc", ltc_hidden=64, ltc_lr=0.001,
                                energy_decay=0.003)
    agent2.arbiter.net.load_state_dict(agent.arbiter.net.state_dict())
    agent2.set_total_cells(world2.cfg.height * world2.cfg.width)
    world2.register_agent("ltc_trained")
    return run_agent(agent2, world2, MAX_STEPS)


def print_table(results):
    print(f"\n{'='*75}")
    print(f"Evola v0.1 Benchmark — {MAX_STEPS} steps, seed={SEED}")
    print(f"{'='*75}")
    print(f"{'Agent':<16} {'Survive':<8} {'Dead@':<8} {'Energy':<10} {'Explore':<9} {'Food':<6} {'Eff/1k':<8}")
    print(f"{'-'*75}")
    for name, r in results.items():
        dead = "-" if r["survived"] else str(r["death_step"])
        print(f"{name:<16} {'YES' if r['survived'] else 'NO':<8} {dead:<8} "
              f"{r['energy_mean']:<10.3f} {r['exploration']:<9.1%} "
              f"{r['food_eaten']:<6} {r['efficiency']:<8.1f}")
    print(f"{'='*75}")


if __name__ == "__main__":
    results = {}
    np.random.seed(SEED)

    print("Testing Random...")
    results["Random"] = test_random()

    print("Testing Rule (MVP0)...")
    results["Rule"] = test_rule()

    print("Testing LTC untrained...")
    results["LTC untrained"] = test_ltc_untrained()

    print("Testing LTC trained (500 pretrain)...")
    results["LTC trained"] = test_ltc_trained(500)

    print_table(results)
