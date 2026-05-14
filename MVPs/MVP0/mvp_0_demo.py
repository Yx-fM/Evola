"""MVP 0 — First Integrated Evola Prototype

Assembles all P0 modules into a running agent:
  08_kunyu (world) + 02_homeostasis (needs) + 10_sensorimotor (mind) + 09_junjian (observer)

Usage:
  python MVPs/mvp_0_demo.py                     # terminal mode, 500 steps
  python MVPs/mvp_0_demo.py --steps 2000         # longer run
  python MVPs/mvp_0_demo.py --render mpl --delay 0.02  # visual mode
  python MVPs/mvp_0_demo.py --render none         # headless, stats only
  python MVPs/mvp_0_demo.py --seed 123            # reproducible
"""

import sys, os, argparse, json, time
from datetime import datetime

_BASE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(os.path.dirname(_BASE))  # MVPs/MVP0 → MVPs → root

# Bridge imports across numeric-prefix folders
sys.path.insert(0, os.path.join(_PROJ, "verification", "08_kunyu", "impl"))
sys.path.insert(0, os.path.join(_PROJ, "verification", "02_homeostasis", "impl"))
sys.path.insert(0, os.path.join(_PROJ, "verification", "09_junjian", "impl"))
sys.path.insert(0, os.path.join(_PROJ, "verification", "10_sensorimotor", "impl"))

import numpy as np
from event_bus import EventBus
from world import KunyuWorld, WorldConfig
from agent import SensorimotorAgent
from logger import JunJian
from mvp_0_config import WORLD, HOMEOSTASIS, ARBITER, RUN
from mvp_0_stats import StatsTracker


def build_world(bus):
    cfg = WorldConfig(**WORLD)
    return KunyuWorld(cfg, bus=bus)


def build_agent(rng):
    return SensorimotorAgent(
        agent_id="evo_001",
        energy_decay=HOMEOSTASIS["energy_decay"],
        safety_rise=HOMEOSTASIS["safety_rise"],
        safety_decay=HOMEOSTASIS["safety_decay"],
        arbiter_params=ARBITER,
        rng=rng,
    )


def save_mind(agent, steps: int):
    mind_dir = os.path.join(_BASE, "minds")
    os.makedirs(mind_dir, exist_ok=True)
    path = os.path.join(mind_dir, f"{agent.agent_id}.mind.evola")
    h = agent.homeo
    data = {
        "format": "evola.mind",
        "version": "mvp0.1",
        "agent_id": agent.agent_id,
        "created_at": datetime.now().isoformat(),
        "saved_step": steps,
        "step_count": agent.step_count,
        "homeostasis": {
            "energy": h.energy.value,
            "novelty_visited_count": h.novelty.visited_count,
            "novelty_total_cells": h.novelty.total_cells,
            "safety": h.safety.value,
        },
        "visited": [[int(p[0]), int(p[1])] for p in agent.visited],
        "arbiter_params": ARBITER,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n[Mind saved] {path}")


def run_terminal(world, agent, stats, max_steps):
    """Fast terminal rendering with ANSI codes."""
    from render_terminal import render

    for step in range(1, max_steps + 1):
        obs = world.get_obs(agent.agent_id)
        action = int(agent.act(obs))
        _, info = world.step(action, agent_id=agent.agent_id)
        agent.observe(info)
        stats.record(step, agent)

        if step % 10 == 0 or step <= 5 or step >= max_steps - 5:
            render(world)
            h = agent.homeo
            drive = h.get_drive()
            print(f"  Step {step:4d} | E={h.energy.value:.2f} D_E={drive.energy:.2f} "
                  f"N={h.novelty.value:.2f} S={h.safety.value:.2f} "
                  f"Visited={h.novelty.visited_count}")


def run_mpl(world, agent, stats, max_steps, delay):
    """Visual matplotlib rendering."""
    import matplotlib.pyplot as plt
    from render_mpl import render_grid, render_obs

    plt.ion()
    fig, (ax_w, ax_o, ax_i) = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Evola MVP 0 — Sensorimotor Agent", fontsize=12)
    info_text = ax_i.text(0.1, 0.5, "", fontsize=9, fontfamily="monospace",
                          verticalalignment="center", transform=ax_i.transAxes)
    ax_i.axis("off")

    for step in range(1, max_steps + 1):
        if not plt.fignum_exists(fig.number):
            break

        obs = world.get_obs(agent.agent_id)
        action = int(agent.act(obs))
        _, info = world.step(action, agent_id=agent.agent_id)
        agent.observe(info)
        stats.record(step, agent)

        if step % 5 == 0 or step <= 3:
            ax_w.clear(); ax_o.clear()
            render_grid(world, ax=ax_w, title=f"Step {step}")
            render_obs(world, ax=ax_o, title="Agent View")
            h = agent.homeo
            drive = h.get_drive()
            lines = [
                f"Step: {step}",
                f"Energy: {h.energy.value:.2f}  Drive: {drive.energy:.2f}",
                f"Novelty: {h.novelty.value:.2f}  Drive: {drive.novelty:.2f}",
                f"Safety:  {h.safety.value:.2f}  Drive: {drive.safety:.2f}",
                f"Visited: {h.novelty.visited_count}",
            ]
            info_text.set_text("\n".join(lines))
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(delay)

    plt.ioff()
    plt.close(fig)


def run_headless(world, agent, stats, max_steps):
    """No rendering, fastest execution."""
    for step in range(1, max_steps + 1):
        obs = world.get_obs(agent.agent_id)
        action = int(agent.act(obs))
        _, info = world.step(action, agent_id=agent.agent_id)
        agent.observe(info)
        stats.record(step, agent)

        if step % 100 == 0:
            h = agent.homeo
            print(f"  Step {step:4d} | E={h.energy.value:.2f} "
                  f"N={h.novelty.value:.2f} S={h.safety.value:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Evola MVP 0")
    parser.add_argument("--steps", type=int, default=RUN["max_steps"])
    parser.add_argument("--render", choices=["terminal", "mpl", "none"],
                        default=RUN["render_mode"])
    parser.add_argument("--delay", type=float, default=RUN["delay"])
    parser.add_argument("--seed", type=int, default=RUN["seed"])
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"=== Evola MVP 0 === seed={args.seed} steps={args.steps}\n")

    # 1. Platform
    bus = EventBus()
    log_dir = os.path.join(_BASE, "logs")
    junjian = None
    if not args.no_log:
        junjian = JunJian(bus, log_dir=log_dir).start()
        print(f"[JunJian] logging to {os.path.basename(junjian.get_log_path())}")

    world = build_world(bus)
    agent = build_agent(rng)
    agent.set_total_cells(world.cfg.height * world.cfg.width)
    world.register_agent(agent.agent_id)

    total_cells = world.cfg.height * world.cfg.width
    stats = StatsTracker(agent.agent_id, total_cells)

    # 2. Main loop
    t0 = time.perf_counter()

    if args.render == "terminal":
        run_terminal(world, agent, stats, args.steps)
    elif args.render == "mpl":
        run_mpl(world, agent, stats, args.steps, args.delay)
    else:
        run_headless(world, agent, stats, args.steps)

    elapsed = time.perf_counter() - t0
    print(f"\n[Done] {args.steps} steps in {elapsed:.1f}s "
          f"({args.steps/elapsed:.0f} steps/s)")

    # 3. Save output
    h = agent.homeo
    print(f"  Energy: {h.energy.value:.2f}  "
          f"Novelty: {h.novelty.value:.2f}  "
          f"Safety:  {h.safety.value:.2f}")
    print(f"  Visited: {h.novelty.visited_count}/{total_cells} cells "
          f"({h.novelty.visited_count/total_cells*100:.1f}%)")

    if not args.no_save:
        save_mind(agent, args.steps)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stat_dir = os.path.join(_BASE, "stats")
        csv_path = os.path.join(stat_dir, f"mvp0_{timestamp}.csv")
        plot_path = os.path.join(stat_dir, f"mvp0_{timestamp}.png")
        stats.save_csv(csv_path)
        stats.save_plot(plot_path)
        print(f"[Stats] {csv_path}")
        print(f"[Plot]  {plot_path}")

    if junjian:
        junjian.stop()

    print("\nMVP 0 complete.")


if __name__ == "__main__":
    main()
