"""Evola MVP 1 Pygame �?LTC agent with full GUI.

Launch: python MVPs/MVP1/evola_pygame.py
"""

import sys, os, json
from datetime import datetime
from collections import deque

_BASE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(os.path.dirname(_BASE))

sys.path.insert(0, os.path.join(_PROJ, "verification", "V0.1", "08_kunyu", "impl"))
sys.path.insert(0, os.path.join(_PROJ, "verification", "V0.1", "02_homeostasis", "impl"))
sys.path.insert(0, os.path.join(_PROJ, "verification", "V0.1", "09_junjian", "impl"))
sys.path.insert(0, os.path.join(_PROJ, "verification", "V0.1", "10_sensorimotor", "impl"))

import numpy as np
import pygame
from event_bus import EventBus
from world import KunyuWorld, WorldConfig
from agent import SensorimotorAgent
from logger import JunJian
from mvp_1_config import WORLD, HOMEOSTASIS, ARBITER

T = {
    "title": "未晞 Evola MVP 1 - LTC",
    "start": "启动", "spawn": "生成", "load": "载入", "run100": "�?00�?, "run": "持续�?,
    "pause": "暂停", "resume": "继续", "extract": "取出", "debug": "调试",
    "stop_btn": "关闭", "quit_btn": "退�?,
    "energy": "能量", "novelty": "好奇", "safety": "安全",
    "visited": "足迹", "events": "事件", "privacy": "�?�?钧鉴·明镜 �?�?,
    "level": "LTC在线学习", "debug_on": "调试�?, "no_agents": "暂无智能�?,
    "start_first": "请先启动平台",
    "platform_started": "MVP1 平台已启�? {}x{}", "spawned": "已生�?{} (LTC)",
    "ran_steps": "已运�?{} �?, "extracted": "已取�? {}", "loaded": "已载�? {}", "loaded": "已载�? {}",
    "platform_stopped": "平台已关�?, "not_found": "未找�?,
    "extract_first": "请先取出智能�?, "already_running": "平台已在运行",
    "already_exists": "同名智能体已存在", "step_label": "步数",
    "agents_label": "智能�?, "fps_label": "FPS",
    "paused_label": "[已暂停]", "idle": "待命�?,
    "exploring": "探索�?..", "ate_food": "吃到了食�?,
    "took_damage": "受到了伤�?, "hit_wall": "撞到了墙", "saw_food": "发现了食�?,
    "starving": "饥饿", "hungry": "饿了", "peckish": "有点�?, "full": "饱足",
    "familiar": "熟悉", "curious": "好奇", "restless": "不安", "bored": "无聊",
    "safe": "安全", "alert": "警觉", "scared": "害�?, "terrified": "恐惧",
}

C_BG = (18, 18, 28); C_PANEL_BG = (26, 26, 38); C_TOOLBAR = (32, 32, 48)
C_BTN = (48, 48, 68); C_BTN_HOVER = (68, 68, 90); C_BTN_ACTIVE = (90, 140, 90)
C_BTN_DISABLED = (38, 38, 52); C_TEXT = (220, 220, 220); C_DIM = (130, 130, 155)
C_TERRAIN = {0: (28, 28, 42), 1: (52, 68, 52), 2: (65, 155, 70), 3: (225, 60, 50), 4: (217, 162, 58)}
C_VISITED = (75, 75, 95); C_GRID_LINE = (22, 22, 36)
C_EVOLA = (217, 162, 58); C_EVOLA_HL = (229, 183, 61)
C_DEBUG = (214, 92, 92); C_PRIVACY = (94, 91, 107); C_BAR_BG = (38, 38, 54)
C_NOVELTY_BAR = (90, 170, 190)
TOOLBAR_H = 36; STATUS_H = 26

PRESETS = {
    "tiny":  {"height": 8,  "width": 10, "n_food": 3, "n_dangers": 1},
    "small": {"height": 12, "width": 15, "n_food": 5, "n_dangers": 2},
    "large": {"height": 30, "width": 40, "n_food": 15, "n_dangers": 5},
}


def _label(value, ts_d):
    for t, k in ts_d.items():
        if value <= t: return T[k]
    return T[list(ts_d.values())[-1]]


class EvolaPygameLTC:
    def __init__(self):
        pygame.init()
        self.platform_running = False
        self.bus = None; self.junjian = None; self.world = None
        self.agents: dict[str, SensorimotorAgent] = {}
        self.debug = False; self.paused = True; self.running = True
        self.auto_step = False; self.run_steps = 0
        self.events: deque = deque(maxlen=12)
        self._last_event_desc = ""
        self._speed = 1          # render every N steps (1=every, 5=every 5th)
        self.mind_dir = os.path.join(_BASE, "minds")
        os.makedirs(self.mind_dir, exist_ok=True)
        self._msg = ""; self._msg_timer = 0
        self._btn_rects: list = []
        self._width = 1000; self._height = 620
        self._world_cfg = dict(WORLD)
        self._world_size = "std"
        self._screen = pygame.display.set_mode((self._width, self._height), pygame.RESIZABLE)
        pygame.display.set_caption(T["title"]); self._clock = pygame.time.Clock()
        self._update_fonts()

    def _load_font(self, size):
        for fp in ["C:\\Windows\\Fonts\\msyh.ttc", "C:\\Windows\\Fonts\\simhei.ttf"]:
            if os.path.exists(fp): return pygame.font.Font(fp, size)
        return pygame.font.Font(None, size)

    def _update_fonts(self):
        sz = max(10, self._height // 50)
        self._font = self._load_font(sz); self._font_sm = self._load_font(max(9, sz - 2))

    def cmd_start(self):
        if self.platform_running: self._msg = T["already_running"]; return
        self.bus = EventBus()
        self.junjian = JunJian(self.bus, log_dir=os.path.join(_BASE, "logs")).start()
        self.world = KunyuWorld(WorldConfig(**self._world_cfg), bus=self.bus)
        self.platform_running = True; self.paused = True
        self._msg = T["platform_started"].format(self.world.cfg.height, self.world.cfg.width)

    def cmd_spawn(self, agent_id=None):
        if not self.platform_running: self._msg = T["start_first"]; return
        agent_id = agent_id or f"evo_{len(self.agents)+1}"
        if agent_id in self.agents: self._msg = T["already_exists"]; return
        agent = SensorimotorAgent(
            agent_id=agent_id, energy_decay=HOMEOSTASIS["energy_decay"],
            arbiter_type=ARBITER["type"], ltc_hidden=ARBITER["ltc_hidden"],
            ltc_lr=ARBITER["ltc_lr"], arbiter_params=ARBITER,
        )
        agent.set_total_cells(self.world.cfg.height * self.world.cfg.width)
        self.world.register_agent(agent.agent_id)
        self.agents[agent_id] = agent
        self._msg = T["spawned"].format(agent_id)

    def cmd_run(self, n):
        if not self.platform_running or not self.agents: return
        for _ in range(n): self._step_all()
        self._msg = T["ran_steps"].format(n)

    def cmd_extract(self, agent_id):
        if agent_id not in self.agents: self._msg = T["not_found"]; return
        agent = self.agents.pop(agent_id); self.world.remove_agent(agent_id)
        path = os.path.join(self.mind_dir, f"{agent_id}.mind.evola")
        agent.save_mind(path)
        self._msg = T["extracted"].format(agent_id)

    def cmd_load(self, agent_id: str):
        if not self.platform_running: self._msg = T["start_first"]; return
        path = os.path.join(self.mind_dir, f"{agent_id}.mind.evola")
        if not os.path.exists(path): self._msg = T["not_found"]; return
        agent = SensorimotorAgent.load_mind(path,
            energy_decay=HOMEOSTASIS["energy_decay"],
            arbiter_params=ARBITER)
        agent.set_total_cells(self.world.cfg.height * self.world.cfg.width)
        self.world.register_agent(agent_id); self.agents[agent_id] = agent
        self._msg = T["loaded"].format(agent_id)

    def cmd_stop(self):
        if self.agents: self._msg = T["extract_first"]; return
        if self.junjian: self.junjian.stop()
        self.platform_running = False; self.world = None; self.junjian = None; self.bus = None
        self.paused = True; self.auto_step = False; self.events.clear()
        self._msg = T["platform_stopped"]

    def _step_all(self):
        for aid, agent in self.agents.items():
            obs = self.world.get_obs(aid)
            action = int(agent.act(obs))
            _, info = self.world.step(action, agent_id=aid)
            agent.observe(info)
            self._record(aid, info)

    def _record(self, aid, info):
        g = info.get("energy_gained", 0); d = info.get("damage_taken", 0)
        c = info.get("collision", False)
        if g > 0: ev = f"{aid} {T['ate_food']}"
        elif d > 0: ev = f"{aid} {T['took_damage']}"
        elif c: ev = f"{aid} {T['hit_wall']}"
        elif info.get("food_visible"): ev = f"{aid} {T['saw_food']}"
        else: ev = f"{aid} {T['exploring']}"
        # Dedup: skip repeated identical events
        if ev == self._last_event_desc:
            return
        self._last_event_desc = ev
        self.events.append((self.world.step_count, ev))

    def _cell_size(self):
        if not self.world: return 24
        return max(8, (self._height - TOOLBAR_H - STATUS_H) // self.world.cfg.height)

    def _draw_btn(self, rect, text, enabled=True, active=False):
        hover = enabled and rect.collidepoint(pygame.mouse.get_pos())
        c = C_BTN_ACTIVE if active else (C_BTN_HOVER if hover else C_BTN)
        if not enabled: c = C_BTN_DISABLED
        r = rect.inflate(-2, -4)
        pygame.draw.rect(self._screen, c, r, border_radius=4)
        t = self._font_sm.render(text, True, C_TEXT if enabled else C_DIM)
        self._screen.blit(t, (r.centerx - t.get_width()//2, r.centery - t.get_height()//2))

    def _draw_bar(self, x, y, w, h, value, color):
        pygame.draw.rect(self._screen, C_BAR_BG, (x, y, w, h))
        fw = int(max(0, min(1, value)) * w)
        if fw > 0: pygame.draw.rect(self._screen, color, (x, y, fw, h))

    def _draw_panel(self):
        if not self.world: return
        cell = self._cell_size()
        world_w = self.world.cfg.width * cell
        panel_x = world_w + 4; panel_w = self._width - panel_x - 4
        if panel_w < 160: return
        px2 = panel_x + 6; py2 = TOOLBAR_H + 6; pw = panel_w - 12
        f = self._font; fs = self._font_sm
        pygame.draw.rect(self._screen, C_PANEL_BG, (panel_x, TOOLBAR_H, panel_w, self._height - TOOLBAR_H - STATUS_H))

        if not self.agents:
            t = f.render(T["no_agents"], True, C_DIM); self._screen.blit(t, (px2, py2)); return

        # Header
        t = f.render(f"Agents: {len(self.agents)}  [{T['level']}]", True, C_EVOLA)
        self._screen.blit(t, (px2, py2)); py2 += 10

        last_ev = self.events[-1][1] if self.events else T["idle"]
        t = fs.render(last_ev, True, C_EVOLA_HL); self._screen.blit(t, (px2, py2)); py2 += 18

        # Each agent compact row
        for aid, ag in self.agents.items():
            h = ag.homeo; nv = len(ag.visited); total = h.novelty.total_cells

            pygame.draw.line(self._screen, (45, 45, 60), (px2, py2), (px2 + pw, py2)); py2 += 4

            pos = self.world.get_agent_pos(aid)
            el = _label(h.energy.value, {0.2: "starving", 0.5: "hungry", 0.8: "peckish", 1.0: "full"})
            t = fs.render(f"{aid} @{pos}  {el}", True, C_EVOLA)
            self._screen.blit(t, (px2, py2)); py2 += 14

            # Compact 3-bar row
            bw = int(pw * 0.28); gap = 4
            self._draw_bar(px2, py2, bw, 4, h.energy.value, C_EVOLA)
            self._draw_bar(px2 + bw + gap, py2, bw, 4, h.novelty.value, C_NOVELTY_BAR)
            self._draw_bar(px2 + 2*(bw + gap), py2, bw, 4, 1 - h.safety.value, (230, 65, 55))
            py2 += 6
            t = fs.render(f"E {h.energy.value:.1f}  N {h.novelty.value:.1f}  S {h.safety.value:.1f}  {T['visited']}: {nv}", True, C_DIM)
            self._screen.blit(t, (px2, py2)); py2 += 14

        # Privacy + Events
        py2 += 4
        plc = C_DEBUG if self.debug else C_PRIVACY
        pltxt = T["debug_on"] if self.debug else T["privacy"]
        pygame.draw.line(self._screen, plc, (px2, py2), (px2 + pw, py2)); py2 += 6
        self._screen.blit(fs.render(pltxt, True, plc), (px2 + 20, py2)); py2 += 20

        log_y = py2
        self._screen.blit(fs.render(T["events"], True, C_DIM), (px2, log_y)); log_y += 16
        evc = {"ate": C_EVOLA, "took": (230, 65, 55), "hit": C_DIM, "saw": (150, 180, 150), "exploring": (150, 180, 150)}
        for _, desc in reversed(list(self.events)[-7:]):
            c = C_DIM
            for k, col in evc.items():
                if k in desc: c = col; break
            self._screen.blit(fs.render(desc, True, c), (px2, log_y)); log_y += 14

        t = fs.render(f"{T['visited']}: {nv}/{total} ({nv/total*100:.1f}%)", True, C_DIM)
    def _draw_grid(self):
        if not self.world: return
        cell = self._cell_size(); w = self.world
        visited_union = set()
        for ag in self.agents.values():
            visited_union |= ag.visited
        for r in range(w.cfg.height):
            for c in range(w.cfg.width):
                color = C_TERRAIN.get(w.terrain[r, c], (0, 0, 0))
                rect = pygame.Rect(c*cell, TOOLBAR_H+r*cell, cell, cell)
                pygame.draw.rect(self._screen, color, rect)
                pygame.draw.rect(self._screen, C_GRID_LINE, rect, 1)
                for aid, ag in self.agents.items():
                    if (r, c) == w.get_agent_pos(aid):
                        pygame.draw.circle(self._screen, C_EVOLA, (c*cell+cell//2, TOOLBAR_H+r*cell+cell//2), max(3, cell//2-1))
                if w.terrain[r, c] == 0 and (r, c) in visited_union:
                    pygame.draw.circle(self._screen, C_VISITED, (c*cell+cell//2, TOOLBAR_H+r*cell+cell//2), max(2, cell//6))

    def _draw_toolbar(self):
        world_w = self.world.cfg.width * self._cell_size() if self.world else 400
        pygame.draw.rect(self._screen, C_TOOLBAR, (0, 0, max(self._width, world_w), TOOLBAR_H))
        self._btn_rects.clear()
        rn = self.platform_running; ha = bool(self.agents)
        btns = [
            ("�?, "sizesel_tiny", not rn),
            ("�?, "sizesel_small", not rn),
            ("�?, "sizesel_std", not rn),
            ("�?, "sizesel_large", not rn),
            (T["start"], "start", not rn), (T["spawn"], "spawn", rn and not self.auto_step),
            (T["load"], "load", rn and not self.auto_step),
            (T["run100"], "run100", rn and not self.auto_step),
            (T["run"], "run", rn and not self.auto_step),
            (f"x{self._speed}", "speed", rn),
            (T["resume"] if (self.auto_step and not self.paused) else T["pause"], "pause", rn),
            (T["extract"], "extract", ha), (T["debug"], "debug", rn),
            (T["stop_btn"], "stop", rn), (T["quit_btn"], "quit", True),
        ]
        x = 4
        for label, cmd, enabled in btns:
            wb = max(60, self._font_sm.size(label)[0] + 16)
            r = pygame.Rect(x, 2, wb, TOOLBAR_H - 4)
            active = (cmd == "debug" and self.debug) or (cmd == "pause" and self.auto_step and not self.paused)
            self._draw_btn(r, label, enabled, active)
            if enabled: self._btn_rects.append((cmd, r))
            x += wb + 4

    def _draw_status(self):
        total_w, total_h = self._width, self._height
        pygame.draw.rect(self._screen, (22, 22, 38), (0, total_h - STATUS_H, total_w, STATUS_H))
        s = f"{T['step_label']}: {self.world.step_count} | " if self.world else f"{T['step_label']}: - | "
        s += f"{T['agents_label']}: {len(self.agents)} | {T['fps_label']}: {int(self._clock.get_fps())}"
        s += " | [Space]" + T["pause"] + " [D]" + T["debug"] + " [Q]" + T["quit_btn"]
        if self.paused: s = T["paused_label"] + " " + s
        t = self._font_sm.render(s, True, C_DIM)
        self._screen.blit(t, (6, total_h - STATUS_H + 6))

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.running = False
                elif event.type == pygame.VIDEORESIZE:
                    self._width, self._height = event.size; self._update_fonts()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE: self.paused = not self.paused
                    elif event.key == pygame.K_d: self.debug = not self.debug
                    elif event.key == pygame.K_q:
                        self.auto_step = False; self.paused = True
                        if not self.platform_running or not self.agents: self.running = False
                    elif event.key == pygame.K_ESCAPE: self.auto_step = False; self.paused = True
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for cmd, rect in self._btn_rects:
                        if rect.collidepoint(event.pos): self._handle_cmd(cmd); break
            if self.auto_step and not self.paused and self.platform_running:
                if self.run_steps > 0:
                    for _ in range(self._speed):
                        if self.run_steps <= 0: break
                        self._step_all(); self.run_steps -= 1
                    if self.run_steps == 0: self.auto_step = False; self.paused = True
            self._screen.fill(C_BG)
            self._draw_toolbar(); self._draw_grid(); self._draw_panel(); self._draw_status()
            if self._msg:
                t = self._font_sm.render(self._msg, True, (255, 255, 200))
                self._screen.blit(t, (self._width//2 - t.get_width()//2, self._height - STATUS_H - 18))
                self._msg_timer -= 1
                if self._msg_timer <= 0: self._msg = ""
            pygame.display.flip(); self._clock.tick(30)
        if self.platform_running:
            for a in list(self.agents): self.cmd_extract(a); self.cmd_stop()
        pygame.quit()

    def _handle_cmd(self, cmd):
        self._msg_timer = 90
        if cmd == "start": self.cmd_start()
        elif cmd.startswith("sizesel_"):
            size = cmd.split("_")[1]
            if size == "std":
                self._world_cfg = dict(WORLD)
            else:
                self._world_cfg = dict(WORLD)
                self._world_cfg.update(PRESETS.get(size, {}))
            self._world_size = size
            self._msg = f"{size} ({self._world_cfg['height']}x{self._world_cfg['width']} F:{self._world_cfg['n_food']})"
        elif cmd == "spawn": self.cmd_spawn()
        elif cmd == "load":
            files = sorted([f for f in os.listdir(self.mind_dir) if f.endswith(".mind.evola")])
            if not files: self._msg = "minds 目录为空"
            else:
                agent_id = files[0].replace(".mind.evola", "")
                self.cmd_load(agent_id)
        elif cmd == "run100": self.auto_step = True; self.paused = False; self.run_steps = 100
        elif cmd == "run": self.auto_step = True; self.paused = False; self.run_steps = 999999
        elif cmd == "pause": self.paused = not self.paused
        elif cmd == "extract":
            for a in list(self.agents): self.cmd_extract(a)
        elif cmd == "debug": self.debug = not self.debug
        elif cmd == "speed":
            speeds = [1, 2, 5, 10]
            idx = speeds.index(self._speed) if self._speed in speeds else 0
            self._speed = speeds[(idx + 1) % len(speeds)]
        elif cmd == "stop": self.cmd_stop()
        elif cmd == "quit":
            if self.agents: self.auto_step = False; self.paused = True
            self.running = False


if __name__ == "__main__":
    EvolaPygameLTC().run()

