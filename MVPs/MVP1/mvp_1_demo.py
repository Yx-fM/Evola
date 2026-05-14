"""MVP 1 �?one-shot demo with LTC agent."""

import sys, os, argparse, json, time
from datetime import datetime

_BASE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(os.path.dirname(_BASE))

sys.path.insert(0, os.path.join(_PROJ, "verification", "V0.1", "08_kunyu", "impl"))
sys.path.insert(0, os.path.join(_PROJ, "verification", "V0.1", "02_homeostasis", "impl"))
sys.path.insert(0, os.path.join(_PROJ, "verification", "V0.1", "09_junjian", "impl"))
sys.path.insert(0, os.path.join(_PROJ, "verification", "V0.1", "10_sensorimotor", "impl"))

import numpy as np
from event_bus import EventBus
from world import KunyuWorld, WorldConfig
from agent import SensorimotorAgent
from mvp_1_config import WORLD, HOMEOSTASIS, ARBITER, RUN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=RUN["max_steps"])
    parser.add_argument("--render", choices=["terminal", "none"], default=RUN["render_mode"])
    parser.add_argument("--seed", type=int, default=RUN["seed"])
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--rule", action="store_true", help="Use RuleArbiter (P0) instead of LTC")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    arb_type = "rule" if args.rule else ARBITER["type"]
    print(f"=== Evola MVP 1 {'(LTC)' if arb_type == 'ltc' else '(Rule)'} === seed={args.seed} steps={args.steps}\n")

    bus = EventBus()
    world = KunyuWorld(WorldConfig(**WORLD), bus=bus)
    agent = SensorimotorAgent(
        agent_id="evo_001", energy_decay=HOMEOSTASIS["energy_decay"],
        arbiter_type=arb_type, ltc_hidden=ARBITER["ltc_hidden"],
        ltc_lr=ARBITER["ltc_lr"], arbiter_params=ARBITER, rng=rng,
    )
    agent.set_total_cells(world.cfg.height * world.cfg.width)
    world.register_agent(agent.agent_id)

    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        obs = world.get_obs(agent.agent_id)
        action = int(agent.act(obs))
        _, info = world.step(action, agent_id=agent.agent_id)
        agent.observe(info)

        if step % 100 == 0:
            h = agent.homeo
            print(f"  Step {step:5d} | E={h.energy.value:.2f} N={h.novelty.value:.2f} S={h.safety.value:.2f}  Visited={len(agent.visited)}")

    elapsed = time.perf_counter() - t0
    h = agent.homeo
    print(f"\n[Done] {args.steps} steps in {elapsed:.1f}s ({args.steps/elapsed:.0f} s/s)")
    print(f"  Energy: {h.energy.value:.2f}  Novelty: {h.novelty.value:.2f}  Safety: {h.safety.value:.2f}")
    print(f"  Visited: {len(agent.visited)}/{world.cfg.height*world.cfg.width} cells ({len(agent.visited)/(world.cfg.height*world.cfg.width)*100:.1f}%)")

    if not args.no_save:
        os.makedirs(os.path.join(_BASE, "minds"), exist_ok=True)
        path = os.path.join(_BASE, "minds", f"{agent.agent_id}.mind.evola")
        data = {"format": "evola.mind", "version": "mvp1.0", "agent_id": agent.agent_id,
                "created_at": datetime.now().isoformat(), "saved_step": args.steps,
                "step_count": agent.step_count,
                "homeostasis": {"energy": h.energy.value, "novelty_visited_count": len(agent.visited),
                                "novelty_total_cells": h.novelty.total_cells, "safety": h.safety.value},
                "visited": [[int(p[0]), int(p[1])] for p in agent.visited],
                "arbiter_type": arb_type}
        with open(path, "w", encoding="utf-8") as f: json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  [Mind saved] {path}")


if __name__ == "__main__":
    main()

