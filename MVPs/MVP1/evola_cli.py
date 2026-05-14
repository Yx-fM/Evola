"""MVP 1 �?CLI with LTC agent."""

import sys, os, json, time, shlex
from datetime import datetime
from collections import deque

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
from logger import JunJian
from mvp_1_config import WORLD, HOMEOSTASIS, ARBITER, RUN

sys.path.insert(0, os.path.join(_PROJ, "MVPs", "MVP0"))
from theme import RICH_THEME as T, EVENT_COLORS

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich import box

console = Console()
_MAX_LOG = 15


def _bar(value, width=16):
    filled = int(max(0, min(1, value)) * width)
    chars = "\u2588" * filled + "\u2591" * (width - filled)
    c = "green" if value > 0.5 else ("yellow" if value > 0.2 else "red")
    return f"[{c}]{chars}[/{c}]"


def _label(value, ts):
    for t, n in ts:
        if value <= t: return n
    return ts[-1][1]


PRESETS = {
    "tiny":  {"height": 8,  "width": 10, "n_food": 3, "n_dangers": 1},
    "small": {"height": 12, "width": 15, "n_food": 5, "n_dangers": 2},
    "large": {"height": 30, "width": 40, "n_food": 15, "n_dangers": 5},
}


class EvolaApp:
    def __init__(self):
        self.platform_running = False
        self.bus = None; self.junjian = None; self.world = None
        self.agents: dict[str, SensorimotorAgent] = {}
        self.debug = False
        self.events: deque = deque(maxlen=_MAX_LOG)
        self._running = True
        self.mind_dir = os.path.join(_BASE, "minds")
        os.makedirs(self.mind_dir, exist_ok=True)
        self._msg = ""
        self._last_event_desc = ""
        self._world_cfg = dict(WORLD)  # copy, mutable

    def cmd_start(self, arg=""):
        if self.platform_running: self._msg = "[yellow]Already running[/yellow]"; return
        cfg = dict(self._world_cfg)

        # Parse custom size: "20x30", "tiny", "large", "f=10", "d=3"
        for part in shlex.split(arg) if arg else []:
            if "x" in part:
                try:
                    w, h = part.split("x"); cfg["width"] = int(w); cfg["height"] = int(h)
                except ValueError: pass
            elif part in PRESETS:
                cfg.update(PRESETS[part])
            elif part.startswith("f="):
                try: cfg["n_food"] = int(part[2:])
                except ValueError: pass
            elif part.startswith("d="):
                try: cfg["n_dangers"] = int(part[2:])
                except ValueError: pass

        self.bus = EventBus()
        self.junjian = JunJian(self.bus, log_dir=os.path.join(_BASE, "logs")).start()
        self.world = KunyuWorld(WorldConfig(**cfg), bus=self.bus)
        self.platform_running = True
        self._msg = f"[green]Platform: {self.world.cfg.height}x{self.world.cfg.width} F:{cfg['n_food']} D:{cfg['n_dangers']}[/green]"

    def cmd_spawn(self, agent_id=None):
        if not self.platform_running: self._msg = "[red]Start first[/red]"; return
        agent_id = agent_id or f"evo_{len(self.agents)+1}"
        if agent_id in self.agents: self._msg = "[red]Exists[/red]"; return
        agent = SensorimotorAgent(
            agent_id=agent_id,
            energy_decay=HOMEOSTASIS["energy_decay"],
            arbiter_type=ARBITER["type"],
            ltc_hidden=ARBITER["ltc_hidden"],
            ltc_lr=ARBITER["ltc_lr"],
            arbiter_params=ARBITER,
        )
        agent.set_total_cells(self.world.cfg.height * self.world.cfg.width)
        self.world.register_agent(agent.agent_id)
        self.agents[agent_id] = agent
        self._msg = f"[green]Spawned {agent_id} (LTC)[/green]"

    def cmd_run(self, n):
        if not self.platform_running or not self.agents: self._msg = "[red]No agents[/red]"; return
        for _ in range(n): self._step_all()
        self._msg = f"[green]Ran {n} steps[/green]"

    def cmd_observe(self, max_steps):
        if not self.platform_running or not self.agents: self._msg = "[red]No agents[/red]"; return
        try:
            import pygame; self._observe_pygame(max_steps)
        except ImportError:
            done = 0
            try:
                while max_steps == 0 or done < max_steps:
                    self._step_all(); done += 1; time.sleep(0.02)
            except KeyboardInterrupt:
                pass
            self._msg = f"[green]Observed {done} steps[/green]"

    def _observe_pygame(self, max_steps):
        import pygame; pygame.init()
        w = self.world; cell = 24
        h, wid = w.cfg.height, w.cfg.width
        total_w = wid * cell + 280; total_h = 36 + h * cell + 26
        screen = pygame.display.set_mode((total_w, total_h), pygame.RESIZABLE)
        pygame.display.set_caption("Evola MVP 1 �?LTC Agent")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 14)
        colors = {0: (28, 28, 42), 1: (55, 70, 55), 2: (70, 160, 75), 3: (230, 65, 55), 4: (217, 162, 58)}
        steps_done = 0; paused = False; running = True
        while running and (max_steps == 0 or steps_done < max_steps):
            for e in pygame.event.get():
                if e.type == pygame.QUIT: running = False
                elif e.type == pygame.KEYDOWN and e.key in (pygame.K_q, pygame.K_ESCAPE): running = False
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE: paused = not paused
            if not paused and running:
                self._step_all(); steps_done += 1
            screen.fill((18, 18, 28))
            for r in range(h):
                for c in range(wid):
                    color = colors.get(w.terrain[r, c], (0, 0, 0))
                    pygame.draw.rect(screen, color, (c*cell, 36+r*cell, cell, cell))
                    pygame.draw.rect(screen, (22, 22, 36), (c*cell, 36+r*cell, cell, cell), 1)
            for aid in self.agents:
                aid_pos = w.get_agent_pos(aid)
                pygame.draw.circle(screen, (217, 162, 58),
                                   (aid_pos[1]*cell+cell//2, 36+aid_pos[0]*cell+cell//2), cell//2-1)
            t = font.render(f"Step: {w.step_count} | [Space]Pause [Q]Quit", True, (200, 200, 200))
            screen.blit(t, (4, 8))
            pygame.display.flip(); clock.tick(30)
        pygame.display.quit(); self._msg = f"[green]Observed {steps_done} steps[/green]"

    def cmd_extract(self, agent_id):
        if agent_id not in self.agents: self._msg = "[red]Not found[/red]"; return
        agent = self.agents.pop(agent_id)
        self.world.remove_agent(agent_id)
        path = os.path.join(self.mind_dir, f"{agent_id}.mind.evola")
        h = agent.homeo
        data = {"format": "evola.mind", "version": "mvp1.0", "agent_id": agent_id,
                "created_at": datetime.now().isoformat(), "saved_step": self.world.step_count,
                "step_count": agent.step_count,
                "homeostasis": {"energy": h.energy.value, "novelty_visited_count": len(agent.visited),
                                "novelty_total_cells": h.novelty.total_cells, "safety": h.safety.value},
                "visited": [[int(p[0]), int(p[1])] for p in agent.visited],
                "arbiter_type": ARBITER["type"], "arbiter_params": ARBITER}
        with open(path, "w", encoding="utf-8") as f: json.dump(data, f, indent=2, ensure_ascii=False)
        self._msg = f"[green]Extracted {agent_id}[/green]"

    def cmd_load(self, agent_id):
        if not self.platform_running: self._msg = "[red]Start first[/red]"; return
        path = os.path.join(self.mind_dir, f"{agent_id}.mind.evola")
        if not os.path.exists(path): self._msg = "[red]Not found[/red]"; return
        with open(path) as f: data = json.load(f)
        agent = SensorimotorAgent(agent_id=agent_id, energy_decay=HOMEOSTASIS["energy_decay"],
                                   arbiter_type=data.get("arbiter_type", "rule"),
                                   ltc_hidden=ARBITER["ltc_hidden"], ltc_lr=ARBITER["ltc_lr"])
        agent.set_total_cells(self.world.cfg.height * self.world.cfg.width)
        agent.step_count = data.get("step_count", 0)
        h = agent.homeo
        h.energy.value = data["homeostasis"]["energy"]
        h.novelty.total_cells = data["homeostasis"]["novelty_total_cells"]
        h.safety.value = data["homeostasis"]["safety"]
        for p in data["visited"]: agent.visited.add((p[0], p[1]))
        h.novelty.set_visited(len(agent.visited))
        self.world.register_agent(agent_id); self.agents[agent_id] = agent
        self._msg = f"[green]Loaded {agent_id}[/green]"

    def cmd_stop(self):
        if self.agents: self._msg = "[red]Extract first[/red]"; return
        if self.junjian: self.junjian.stop()
        self.platform_running = False; self.world = None; self.junjian = None; self.bus = None
        self._msg = "[green]Stopped[/green]"

    def _step_all(self):
        for aid, agent in self.agents.items():
            obs = self.world.get_obs(aid)
            action = int(agent.act(obs))
            _, info = self.world.step(action, agent_id=aid)
            agent.observe(info)
            g = info.get("energy_gained", 0); d = info.get("damage_taken", 0)
            c = info.get("collision", False)
            if g > 0: ev = f"{aid} ate food"
            elif d > 0: ev = f"{aid} took damage"
            elif c: ev = f"{aid} hit wall"
            elif info.get("food_visible"): ev = f"{aid} saw food"
            else: ev = f"{aid} exploring..."
            if ev != getattr(self, '_last_event_desc', ""):
                self._last_event_desc = ev
                self.events.append((self.world.step_count, ev))

    def _build_world_panel(self):
        if not self.platform_running: return Panel("[dim]World not started[/dim]", title="Kunyu", border_style=T["kunyu_border"])
        grid = self.world.get_full_grid(); h, w = grid.shape
        visited = set()
        for ag in self.agents.values():
            visited |= ag.visited
        lines = []
        for r in range(h):
            row = ""
            for c in range(w):
                cell = grid[r, c]; pos = (r, c)
                if cell == 1: row += f"[{T['grid_wall']}]\u2593[/{T['grid_wall']}]"
                elif cell == 2: row += f"[{T['grid_food']}]F[/{T['grid_food']}]"
                elif cell == 3: row += f"[{T['grid_danger']}]X[/{T['grid_danger']}]"
                elif cell == 4: row += f"[{T['grid_agent']}]A[/{T['grid_agent']}]"
                elif pos in visited: row += f"[{T['grid_visited']}]\u00b7[/{T['grid_visited']}]"
                else: row += f"[{T['grid_empty']}]\u00b7[/{T['grid_empty']}]"
            lines.append(Text.from_markup(row))
        title = f"[bold {T['kunyu_border']}]MVP 1 Kunyu {h}x{w}  Step {self.world.step_count}[/bold {T['kunyu_border']}]"
        return Panel(Text("\n").join(lines), title=title, border_style=T["kunyu_border"], box=box.HEAVY)

    def _build_log_panel(self):
        if not self.events: inner = Text("[dim]No events[/dim]")
        else:
            lns = [f"[{T['junjian_dim']}]{s:4d}[/{T['junjian_dim']}]  {ev}" for s, ev in reversed(list(self.events)[-14:])]
            inner = Text.from_markup("\n".join(lns))
        title = f"[bold {T['junjian_color']}]JunJian Log[/bold {T['junjian_color']}]"
        return Panel(inner, title=title, border_style=T["debug_border"] if self.debug else T["junjian_color"], box=box.HEAVY)

    def _build_agent_panel(self):
        if not self.agents: return Panel(Text("[dim]No agents[/dim]"), title="Evola", border_style=T["evola_color"])
        info = Text()
        info.append(f"[italic]Agents: {len(self.agents)} (LTC)[/italic]\n\n")
        for aid, ag in self.agents.items():
            h = ag.homeo; nv = len(ag.visited); total = h.novelty.total_cells
            el = _label(h.energy.value, [(0.2, "starving"), (0.5, "hungry"), (0.8, "peckish"), (1.0, "full")])
            info.append(f"[bold {T['evola_color']}]{aid}[/bold {T['evola_color']}] [dim]@{self.world.get_agent_pos(aid)}[/dim] {el}\n")
            info.append(f"{_bar(h.energy.value, 12)}  {T['visited']}: {nv}/{total}\n\n")
        pl = T["debug_border"] if self.debug else T["privacy_line"]
        info.append(f"[{pl}]{'DEBUG ON' if self.debug else '- - JunJian Mirror - -'}[/{pl}]")
        title = f"[bold {T['evola_color']}]Evola (LTC)[/bold {T['evola_color']}]"
        border = T["debug_border"] if self.debug else T["evola_color"]
        return Panel(info, title=title, border_style=border, box=box.HEAVY)

    def build_dashboard(self):
        root = Layout()
        root.split_column(Layout(name="header", size=3), Layout(name="body"), Layout(name="help", size=3))
        hdr = Text()
        hdr.append(" MVP 1 �?LTC Agent ", style=f"bold white on {T['kunyu_border'].split()[-1]}")
        if self.platform_running:
            hdr.append(f"  Kunyu: {self.world.cfg.height}x{self.world.cfg.width}  Step: {self.world.step_count}")
        else: hdr.append("  Platform not started  Type [bold]start[/bold]")
        if self._msg: hdr.append(f"  {self._msg}")
        root["header"].update(Panel(hdr, box=box.HEAVY))
        body = Table(show_header=False, expand=True, box=None, padding=(0, 1))
        body.add_column(ratio=4); body.add_column(ratio=3); body.add_column(ratio=3)
        body.add_row(self._build_world_panel(), self._build_log_panel(), self._build_agent_panel())
        root["body"].update(body)
        h = Text(); s = "bold yellow"
        h.append("[s]tart ", style=s); h.append("[sp]awn ", style=s); h.append("[r]un ", style=s)
        h.append("[o]bserve ", style=s); h.append("[e]xtract ", style=s); h.append("[d]ebug ", style=s)
        h.append("[q]uit")
        root["help"].update(Panel(h, box=box.SIMPLE))
        return root

    def run(self):
        console.clear(); console.print(self.build_dashboard())
        while self._running:
            try: cmd_line = console.input("[bold cyan]evola>[/bold cyan] ")
            except (EOFError, KeyboardInterrupt): self._running = False; break
            if not cmd_line.strip(): continue
            parts = shlex.split(cmd_line); cmd = parts[0].lower(); args = parts[1:] if len(parts) > 1 else []; arg = args[0] if args else ""
            self._msg = ""
            if cmd in ("s", "start"): self.cmd_start(" ".join(args))
            elif cmd in ("sp", "spawn"): self.cmd_spawn(arg)
            elif cmd in ("r", "run"):
                try: self.cmd_run(int(arg) if arg else 1)
                except ValueError: self._msg = "[red]run <number>[/red]"
            elif cmd in ("o", "observe"):
                try: self.cmd_observe(int(arg) if arg else 0)
                except ValueError: self._msg = "[red]observe <number>[/red]"
            elif cmd in ("e", "extract"): self.cmd_extract(arg)
            elif cmd in ("lo", "load"): self.cmd_load(arg)
            elif cmd == "d": self.debug = not self.debug; self._msg = f"[yellow]Debug {'ON' if self.debug else 'OFF'}[/yellow]"
            elif cmd in ("stop",): self.cmd_stop()
            elif cmd in ("q", "quit"):
                if self.platform_running and self.agents: self._msg = "[red]Extract first[/red]"
                else:
                    if self.platform_running: self.cmd_stop()
                    self._running = False
            elif cmd: self._msg = f"[red]Unknown: {cmd}[/red]"
            console.clear(); console.print(self.build_dashboard())


if __name__ == "__main__":
    EvolaApp().run()

