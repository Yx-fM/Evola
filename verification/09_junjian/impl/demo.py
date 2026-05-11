"""Demo: KunyuWorld + JunJian — 孪生系统

Modes:
  --mode manual   → WASD keyboard control (dev testing)
  --mode auto     → RandomAgent runs autonomously
  --mode auto --agent <id>  → custom agent loaded by id (future)

Architecture:
  08_kunyu (world) + 09_junjian (observer) = platform
  Agent is "loaded" into the world, carries own memory internally
"""

import sys, os, argparse

_BASE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_BASE, "..", "08_kunyu", "impl"))
sys.path.insert(0, os.path.join(_BASE, "..", "09_junjian", "impl"))

import time
import matplotlib.pyplot as plt

from entities import Action
from event_bus import EventBus
from world import KunyuWorld, WorldConfig
from agent_interface import RandomAgent
from render_mpl import render_grid, render_obs
from logger import JunJian


def run_manual(world, junjian, fig, ax_world, ax_obs, ax_info):
    """Human controls agent via WASD."""
    info_text = ax_info.text(0.1, 0.5, "", fontsize=9, fontfamily="monospace",
                             verticalalignment="center", transform=ax_info.transAxes)
    ax_info.axis("off")

    KEY_MAP = {
        "w": Action.UP, "s": Action.DOWN, "a": Action.LEFT, "d": Action.RIGHT,
        "e": Action.EAT, " ": Action.WAIT,
    }

    def on_key(event):
        if event.key == "q":
            junjian.stop()
            plt.close(fig)
            return
        action = KEY_MAP.get(event.key)
        if action is None:
            return

        _, info = world.step(action)
        _redraw(world, ax_world, ax_obs, info_text, info, junjian)

    fig.canvas.mpl_connect("key_press_event", on_key)
    _redraw(world, ax_world, ax_obs, info_text, {}, junjian)
    plt.tight_layout()
    plt.show()
    junjian.stop()


def run_auto(world, agent, junjian, fig, ax_world, ax_obs, ax_info, max_steps=500, delay=0.1):
    """Agent runs autonomously: world → obs → agent.act → world.step → repeat."""
    info_text = ax_info.text(0.1, 0.5, "", fontsize=9, fontfamily="monospace",
                             verticalalignment="center", transform=ax_info.transAxes)
    ax_info.axis("off")

    agent_id = agent.agent_id
    info = {}

    for _ in range(max_steps):
        if not plt.fignum_exists(fig.number):
            break

        obs = world.get_obs(agent_id)
        action = Action(agent.act(obs))
        _, info = world.step(action, agent_id=agent_id)
        agent.observe(info)

        _redraw(world, ax_world, ax_obs, info_text, info, junjian, agent)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(delay)

    junjian.stop()
    print(f"Auto mode finished after {world.step_count} steps")


def _redraw(world, ax_world, ax_obs, info_text, info, junjian, agent=None):
    ax_world.clear()
    ax_obs.clear()
    render_grid(world, ax=ax_world, title="Kunyu (full map)", show_vision=True)
    render_obs(world, ax=ax_obs, title="Agent View")

    state = world.get_state()
    lines = [
        f"Step: {state['step']}",
        f"Agents: {state['n_agents']}  Pos: {state['agent_pos']}",
        f"Visited: {state['n_visited']} cells",
        f"Energy gained: {info.get('energy_gained', 0):.1f}",
        f"Damage taken: {info.get('damage_taken', 0):.1f}",
        f"",
        f"Log: {os.path.basename(junjian.get_log_path())}",
    ]
    if hasattr(agent, 'step_count'):
        lines.insert(1, f"Agent: {agent.agent_id}  Steps: {agent.step_count}")
    info_text.set_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["manual", "auto"], default="manual")
    parser.add_argument("--steps", type=int, default=500, help="max steps in auto mode")
    parser.add_argument("--delay", type=float, default=0.05, help="seconds between steps")
    args = parser.parse_args()

    bus = EventBus()
    junjian = JunJian(bus, log_dir=os.path.join(_BASE, "logs")).start()

    cfg = WorldConfig(height=15, width=20, vision_range=2)
    world = KunyuWorld(cfg, bus=bus)

    fig, (ax_world, ax_obs, ax_info) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Kunyu + JunJian  [{args.mode.upper()} mode]  Q=quit", fontsize=11)

    if args.mode == "manual":
        world.register_agent("human")
        run_manual(world, junjian, fig, ax_world, ax_obs, ax_info)
    else:
        agent = RandomAgent(agent_id="proto")
        world.register_agent(agent.agent_id)
        run_auto(world, agent, junjian, fig, ax_world, ax_obs, ax_info,
                 max_steps=args.steps, delay=args.delay)


if __name__ == "__main__":
    main()
