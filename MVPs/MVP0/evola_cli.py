"""Evola CLI �?钧鉴·观察�?
Persistent rich dashboard. Three-column layout:
  World window (40%) | Event log (30%) | Agent panel (30%)

Launch: python MVPs/MVP0/evola_cli.py
"""

import sys, os, json, time, shlex
from collections import deque
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
from logger import JunJian
from mvp_0_config import WORLD, HOMEOSTASIS, ARBITER
from theme import RICH_THEME as T, EVENT_COLORS

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich import box

console = Console()
_TERRAIN_CHAR = {0: " ", 1: "�?, 2: "F", 3: "X", 4: "A"}
_VISITED_CHAR = "·"
_MAX_LOG = 15


def _bar(value: float, width: int = 16) -> str:
    filled = int(max(0, min(1, value)) * width)
    chars = "�? * filled + "�? * (width - filled)
    if value > 0.5:
        c = "green"
    elif value > 0.2:
        c = "yellow"
    else:
        c = "red"
    return f"[{c}]{chars}[/{c}]"


def _label(value: float, ts: list[tuple[float, str]]) -> str:
    for t, n in ts:
        if value <= t:
            return n
    return ts[-1][1]


class EvolaApp:
    def __init__(self):
        self.platform_running = False
        self.bus = None
        self.junjian = None
        self.world = None
        self.agents: dict[str, SensorimotorAgent] = {}
        self.debug = False
        self.events: deque = deque(maxlen=_MAX_LOG)
        self._running = True
        self.mind_dir = os.path.join(_BASE, "minds")
        os.makedirs(self.mind_dir, exist_ok=True)
        self._msg = ""

    # ── Commands ─────────────────────────────

    def cmd_start(self):
        if self.platform_running:
            self._msg = "[yellow]Already running[/yellow]"; return
        self.bus = EventBus()
        self.junjian = JunJian(self.bus, log_dir=os.path.join(_BASE, "logs")).start()
        self.world = KunyuWorld(WorldConfig(**WORLD), bus=self.bus)
        self.platform_running = True
        self._msg = f"[green]Platform started: {self.world.cfg.height}x{self.world.cfg.width}[/green]"

    def cmd_spawn(self, agent_id: str):
        if not self.platform_running: self._msg = "[red]Start first[/red]"; return
        agent_id = agent_id or f"evo_{len(self.agents)+1}"
        if agent_id in self.agents: self._msg = f"[red]Exists[/red]"; return
        agent = SensorimotorAgent(agent_id=agent_id, energy_decay=HOMEOSTASIS["energy_decay"],
                                   safety_rise=HOMEOSTASIS["safety_rise"],
                                   safety_decay=HOMEOSTASIS["safety_decay"],
                                   arbiter_params=ARBITER)
        agent.set_total_cells(self.world.cfg.height * self.world.cfg.width)
        self.world.register_agent(agent.agent_id)
        self.agents[agent_id] = agent
        self._msg = f"[green]Spawned {agent_id}[/green]"

    def cmd_run(self, n: int):
        if not self.platform_running or not self.agents: self._msg = "[red]No agents[/red]"; return
        for _ in range(n):
            self._step_all()
        self._msg = f"[green]Ran {n} steps[/green]"

    def cmd_observe(self, max_steps: int):
        if not self.platform_running or not self.agents: self._msg = "[red]No agents[/red]"; return
        try:
            import pygame
            self._observe_pygame(max_steps)
        except ImportError:
            label = f"{max_steps}" if max_steps else "until Ctrl+C"
            self._msg = f"[yellow]Observing {label}...[/yellow]"
            try:
                done = 0
                while max_steps == 0 or done < max_steps:
                    self._step_all()
                    done += 1
                    time.sleep(0.02)
            except KeyboardInterrupt:
                pass
            self._msg = f"[green]Observed {done} steps[/green]"

    def _observe_pygame(self, max_steps: int):
        import pygame
        pygame.init()

        w = self.world
        cell = 24  # pixel per cell
        h, wid = w.cfg.height, w.cfg.width
        bar = 40
        screen = pygame.display.set_mode((wid * cell, h * cell + bar))
        pygame.display.set_caption(f"Kunyu World �?evola_cli observe ({len(self.agents)} agents)")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("monospace", 13, bold=True)
        small = pygame.font.SysFont("monospace", 11)

        colors = {
            0: (40, 40, 50), 1: (60, 80, 60), 2: (76, 175, 80),
            3: (244, 67, 54), 4: (217, 162, 58),
        }
        visited_c = (100, 100, 120)
        labels = {0: "", 1: "", 2: "F", 3: "X", 4: "A"}

        steps_done = 0
        paused = False
        running = True

        while running and (max_steps == 0 or steps_done < max_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused

            if not paused and running:
                self._step_all()
                steps_done += 1

            screen.fill((20, 20, 30))

            # Draw terrain
            grid = w.get_full_grid()
            visited = self.agents[next(iter(self.agents))].visited if self.agents else set()
            for r in range(h):
                for c in range(wid):
                    cell_t = grid[r, c]
                    color = colors.get(cell_t, (0, 0, 0))
                    rect = pygame.Rect(c * cell, r * cell, cell, cell)
                    pygame.draw.rect(screen, color, rect)
                    # Grid lines
                    pygame.draw.rect(screen, (30, 30, 40), rect, 1)

                    # Visited overlay
                    if cell_t == 0 and (r, c) in visited:
                        pygame.draw.circle(screen, visited_c, (c * cell + cell // 2, r * cell + cell // 2), 3)

                    # Labels
                    label = labels.get(cell_t, "")
                    if label:
                        t = font.render(label, True, (255, 255, 255))
                        tx = c * cell + (cell - t.get_width()) // 2
                        ty = r * cell + (cell - t.get_height()) // 2
                        screen.blit(t, (tx, ty))

            # Status bar
            status = f"Step: {w.step_count} | "
            for aid, ag in self.agents.items():
                hh = ag.homeo
                status += f"{aid} E={hh.energy.value:.2f} "
            status += "| [Space]=Pause [Q]=Quit"
            if paused:
                status = "[PAUSED] " + status
            t = small.render(status, True, (220, 220, 220))
            screen.blit(t, (5, h * cell + 10))

            pygame.display.flip()
            clock.tick(30 if not paused else 10)

        pygame.display.quit()
        self._msg = f"[green]Observed {steps_done} steps[/green]"

    def cmd_extract(self, agent_id: str):
        if agent_id not in self.agents: self._msg = f"[red]Not found[/red]"; return
        agent = self.agents.pop(agent_id)
        self.world.remove_agent(agent_id)
        path = os.path.join(self.mind_dir, f"{agent_id}.mind.evola")
        h = agent.homeo
        data = {
            "format": "evola.mind", "version": "mvp0.1", "agent_id": agent_id,
            "created_at": datetime.now().isoformat(),
            "saved_step": self.world.step_count, "step_count": agent.step_count,
            "homeostasis": {"energy": h.energy.value, "novelty_visited_count": len(agent.visited),
                            "novelty_total_cells": h.novelty.total_cells, "safety": h.safety.value},
            "visited": [[int(p[0]), int(p[1])] for p in agent.visited],
            "arbiter_params": ARBITER,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self._msg = f"[green]Extracted {agent_id} -> {os.path.basename(path)}[/green]"

    def cmd_load(self, agent_id: str):
        if not self.platform_running: self._msg = "[red]Start first[/red]"; return
        path = os.path.join(self.mind_dir, f"{agent_id}.mind.evola")
        if not os.path.exists(path): self._msg = f"[red]File not found[/red]"; return
        with open(path) as f: data = json.load(f)
        agent = SensorimotorAgent(agent_id=agent_id, energy_decay=HOMEOSTASIS["energy_decay"],
                                   arbiter_params=ARBITER)
        agent.set_total_cells(self.world.cfg.height * self.world.cfg.width)
        agent.step_count = data.get("step_count", 0)
        h = agent.homeo
        h.energy.value = data["homeostasis"]["energy"]
        h.novelty.total_cells = data["homeostasis"]["novelty_total_cells"]
        h.safety.value = data["homeostasis"]["safety"]
        for p in data["visited"]: agent.visited.add((p[0], p[1]))
        h.novelty.set_visited(len(agent.visited))
        self.world.register_agent(agent_id)
        self.agents[agent_id] = agent
        self._msg = f"[green]Loaded {agent_id}[/green]"

    def cmd_stop(self):
        if self.agents: self._msg = "[red]Extract agents first[/red]"; return
        if self.junjian: self.junjian.stop()
        self.platform_running = False; self.world = None
        self.junjian = None; self.bus = None
        self._msg = "[green]Platform stopped[/green]"

    # ── Internal ─────────────────────────────

    def _step_all(self):
        for aid, agent in self.agents.items():
            obs = self.world.get_obs(aid)
            action = int(agent.act(obs))
            _, info = self.world.step(action, agent_id=aid)
            agent.observe(info)
            self._record_event(aid, action, info)

    def _record_event(self, aid, action, info):
        s = self.world.step_count
        if info.get("energy_gained", 0) > 0:
            self.events.append((s, "eat", f"{aid} 吃到食物"))
        elif info.get("damage_taken", 0) > 0:
            self.events.append((s, "danger", f"{aid} 遭遇危险"))
        elif info.get("collision"):
            self.events.append((s, "move", f"{aid} 撞墙"))
        elif info.get("food_visible"):
            self.events.append((s, "explore", f"{aid} 发现食物"))
        else:
            self.events.append((s, "explore", f"{aid} 探索�?.."))
        if len(self.events) > _MAX_LOG:
            self.events.popleft()

    def _agent_act_str(self, aid):
        if not self.events: return "�?
        last = self.events[-1]
        return last[2]

    # ── Dashboard builders ───────────────────

    def _build_world_panel(self) -> Panel:
        if not self.platform_running or not self.world:
            return Panel("[dim]坤舆未启动[/dim]", title="世界窗口", border_style=T["kunyu_border"])

        grid = self.world.get_full_grid()
        h, w = grid.shape
        visited = self.agents[next(iter(self.agents))].visited if self.agents else set()
        lines = []
        for r in range(h):
            row = ""
            for c in range(w):
                cell = grid[r, c]
                pos = (r, c)
                if cell == 1:
                    row += f"[{T['grid_wall']}]▓[/{T['grid_wall']}]"
                elif cell == 2:
                    row += f"[{T['grid_food']}]F[/{T['grid_food']}]"
                elif cell == 3:
                    row += f"[{T['grid_danger']}]X[/{T['grid_danger']}]"
                elif cell == 4:
                    row += f"[{T['grid_agent']}]A[/{T['grid_agent']}]"
                elif pos in visited:
                    row += f"[{T['grid_visited']}]·[/{T['grid_visited']}]"
                else:
                    row += f"[{T['grid_empty']}]·[/{T['grid_empty']}]"
            lines.append(Text.from_markup(row))

        s = self.world.step_count
        title = f"[bold {T['kunyu_border']}]坤舆 {h}x{w}  Step {s}[/bold {T['kunyu_border']}]"
        inner = Text("\n").join(lines)
        return Panel(inner, title=title, border_style=T["kunyu_border"], box=box.HEAVY)

    def _build_log_panel(self) -> Panel:
        if not self.events:
            inner = Text("[dim]事件日志为空[/dim]")
        else:
            lines = []
            for step, etype, desc in self.events:
                color = T.get(EVENT_COLORS.get(etype, "event_move"), "white")
                lines.append(f"[{T['junjian_dim']}]{step:4d}[/{T['junjian_dim']}]  [{color}]{desc}[/{color}]")
            inner = Text.from_markup("\n".join(reversed(lines)))

        title = f"[bold {T['junjian_color']}]钧鉴·日志[/bold {T['junjian_color']}]"
        border = T["debug_border"] if self.debug else T["junjian_title"]
        return Panel(inner, title=title, border_style=border, box=box.HEAVY)

    def _build_agent_panel(self) -> Panel:
        if not self.agents:
            inner = Text("[dim]无智能体[/dim]")
            title = f"[bold {T['evola_color']}]未晞[/bold {T['evola_color']}]"
            border = T["debug_border"] if self.debug else T["evola_title"]
            return Panel(inner, title=title, border_style=border, box=box.HEAVY)

        aid = next(iter(self.agents))
        agent = self.agents[aid]
        h = agent.homeo
        d = h.get_drive()
        nv = len(agent.visited)
        total = h.novelty.total_cells

        info = Text()
        info.append(f"[bold {T['evola_color']}]{aid}[/bold {T['evola_color']}]  "
                    f"[dim]@{self.world.get_agent_pos(aid)}[/dim]\n\n")

        status = self._agent_act_str(aid)
        info.append(f"状�? [italic {T['evola_highlight']}]{status}[/italic {T['evola_highlight']}]\n\n")

        # Energy
        label = _label(h.energy.value, [(0.2, "starving"), (0.5, "hungry"), (0.8, "peckish"), (1.0, "full")])
        info.append(f"能量 {_bar(h.energy.value)} ")
        info.append(f"[bold]{h.energy.value:.2f}[/bold] " if self.debug else f"[bold]{label}[/bold] ")
        if self.debug:
            info.append(f"[dim]drive:{d.energy:.2f}[/dim]")
        info.append("\n")

        # Novelty
        label = _label(h.novelty.value, [(0.3, "familiar"), (0.5, "curious"), (0.8, "restless"), (1.0, "bored")])
        info.append(f"好奇 {_bar(h.novelty.value)} ")
        info.append(f"[bold]{h.novelty.value:.2f}[/bold] " if self.debug else f"[bold]{label}[/bold] ")
        if self.debug:
            info.append(f"[dim]drive:{d.novelty:.2f}[/dim]")
        info.append("\n")

        # Safety
        label = _label(h.safety.value, [(0.2, "safe"), (0.5, "alert"), (0.8, "scared"), (1.0, "terrified")])
        info.append(f"安全 {_bar(1 - h.safety.value)} ")
        info.append(f"[bold]{h.safety.value:.2f}[/bold] " if self.debug else f"[bold]{label}[/bold] ")
        if self.debug:
            info.append(f"[dim]drive:{d.safety:.2f}[/dim]")
        info.append("\n\n")

        info.append(f"[dim]足迹: {nv}/{total} ({nv/total*100:.1f}%)[/dim]\n")

        # Privacy line
        if self.debug:
            info.append(f"\n[{T['debug_border']}]━━�?内部状态可�?━━━[/{T['debug_border']}]")
        else:
            info.append(f"\n[{T['privacy_line']}]─ ─ 钧鉴·明镜 ─ ─[/{T['privacy_line']}]")

        title = f"[{T['evola_color']}]未晞[/{T['evola_color']}]"
        border = T["debug_border"] if self.debug else T["evola_title"]
        return Panel(info, title=title, border_style=border, box=box.HEAVY)

    def build_dashboard(self):
        root = Layout()
        root.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="help", size=3),
        )

        # Header
        if self.platform_running and self.world:
            hdr = Text()
            hdr.append(" 钧鉴·观察�?", style=f"bold white on {T['kunyu_border'].split()[-1]}")
            hdr.append(f"  坤舆: {self.world.cfg.height}x{self.world.cfg.width}"
                       f"  Step: {self.world.step_count}")
            hdr.append(f"  智能�? {len(self.agents)}")
            if self._msg:
                hdr.append(f"  {self._msg}")
            root["header"].update(Panel(hdr, box=box.HEAVY))
        else:
            hdr = Text()
            hdr.append(" 钧鉴·观察�?", style=f"bold white on {T['kunyu_border'].split()[-1]}")
            hdr.append("  平台未启�? 输入 [bold]start[/bold] 开�?)
            root["header"].update(Panel(hdr, box=box.HEAVY))

        # Body: 3 columns
        body_table = Table(show_header=False, expand=True, box=None, padding=(0, 1))
        body_table.add_column(ratio=4)   # world
        body_table.add_column(ratio=3)   # log
        body_table.add_column(ratio=3)   # agent
        body_table.add_row(
            self._build_world_panel(),
            self._build_log_panel(),
            self._build_agent_panel(),
        )
        root["body"].update(body_table)

        # Help bar
        help_text = Text()
        s = "bold yellow"
        help_text.append("[s]tart ", style=s)
        help_text.append("[sp]awn ", style=s); help_text.append("[r]un <n> ", style=s)
        help_text.append("[o]bserve ", style=s); help_text.append("[e]xtract <id> ", style=s)
        help_text.append("[lo]ad <id> ", style=s); help_text.append("[st]op ", style=s)
        if self.debug:
            help_text.append("[d]ebug ON ", style=f"bold {T['debug_border']}")
        else:
            help_text.append("[d]ebug ", style=s)
        help_text.append("[q]uit")
        root["help"].update(Panel(help_text, box=box.SIMPLE))

        return root

    def run(self):
        console.clear()
        console.print(self.build_dashboard())

        while self._running:
            try:
                cmd_line = console.input("[bold cyan]evola>[/bold cyan] ")
            except (EOFError, KeyboardInterrupt):
                self._running = False
                break

            if not cmd_line.strip():
                continue

            parts = shlex.split(cmd_line)
            cmd = parts[0].lower()
            args = parts[1:] if len(parts) > 1 else []
            arg = args[0] if args else ""
            self._msg = ""

            if cmd in ("s", "start"):   self.cmd_start()
            elif cmd in ("sp", "spawn"): self.cmd_spawn(arg)
            elif cmd in ("r", "run"):
                try: self.cmd_run(int(arg) if arg else 1)
                except ValueError: self._msg = "[red]run <number>[/red]"
            elif cmd in ("o", "observe"):
                try: self.cmd_observe(int(arg) if arg else 0)
                except ValueError: self._msg = "[red]observe <number>[/red]"
            elif cmd in ("e", "extract"): self.cmd_extract(arg)
            elif cmd in ("lo", "load"): self.cmd_load(arg)
            elif cmd == "d":
                self.debug = not self.debug
                self._msg = f"[yellow]Debug {'ON' if self.debug else 'OFF'}[/yellow]"
            elif cmd in ("stop",): self.cmd_stop()
            elif cmd in ("q", "quit"):
                if self.platform_running and self.agents:
                    self._msg = "[red]Extract first (quit -f to force)[/red]"
                else:
                    if arg == "-f" and self.platform_running and self.agents:
                        for a in list(self.agents): self.cmd_extract(a)
                    if self.platform_running: self.cmd_stop()
                    self._running = False
            elif cmd:
                self._msg = f"[red]Unknown: {cmd}[/red]"

            console.clear()
            console.print(self.build_dashboard())


if __name__ == "__main__":
    EvolaApp().run()

