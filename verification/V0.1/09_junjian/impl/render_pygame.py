"""Pygame renderer - 60fps interactive window with keyboard control."""

import sys
import numpy as np

try:
    import pygame
except ImportError:
    pygame = None


CELL_SIZE = 30
GRID_COLORS = {
    0: (245, 245, 220),  # EMPTY - beige
    1: (74, 74, 74),      # WALL - dark gray
    2: (76, 175, 80),     # FOOD - green
    3: (244, 67, 54),     # DANGER - red
    4: (33, 150, 243),    # AGENT - blue
}
GRID_LABELS = {0: "", 1: "", 2: "F", 3: "X", 4: "A"}

_VISION_COLOR = (255, 235, 59, 60)


class PygameRenderer:
    def __init__(self, world, fps: int = 30, auto_step: bool = True):
        if pygame is None:
            raise ImportError("pygame not installed: pip install pygame")

        self.world = world
        self.fps = fps
        self.auto_step = auto_step
        self._init_window()

    def _init_window(self):
        pygame.init()
        h, w = self.world.cfg.height, self.world.cfg.width
        self.screen = pygame.display.set_mode((w * CELL_SIZE, h * CELL_SIZE + 40))
        pygame.display.set_caption(f"Kunyu World - Step {self.world.step_count}")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 14, bold=True)
        self.small_font = pygame.font.SysFont("monospace", 11)

    def step(self, action):
        self.world.step(action)

    def _draw_grid(self):
        grid = self.world.get_full_grid()
        h, w = grid.shape
        for r in range(h):
            for c in range(w):
                cell = grid[r, c]
                color = GRID_COLORS.get(cell, (0, 0, 0))
                rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)
                label = GRID_LABELS.get(cell, "")
                if label:
                    text = self.font.render(label, True, (255, 255, 255))
                    x = c * CELL_SIZE + CELL_SIZE // 2 - text.get_width() // 2
                    y = r * CELL_SIZE + CELL_SIZE // 2 - text.get_height() // 2
                    self.screen.blit(text, (x, y))

        R = self.world.cfg.vision_range
        ar, ac = self.world.agent_pos
        vision_rect = pygame.Rect(
            (ac - R) * CELL_SIZE, (ar - R) * CELL_SIZE,
            (2 * R + 1) * CELL_SIZE, (2 * R + 1) * CELL_SIZE,
        )
        pygame.draw.rect(self.screen, _VISION_COLOR, vision_rect, 2)

    def _draw_status(self):
        h = self.world.cfg.height
        state = self.world.get_state()
        text = self.small_font.render(
            f"Step: {state['step']}  Pos: {state['agent_pos']}  "
            f"Visited: {state['n_visited']}  [Space]Pause [Arrow]Move [R]Reset [Q]Quit",
            True, (255, 255, 255),
        )
        self.screen.blit(text, (5, h * CELL_SIZE + 8))

    def render(self):
        self.screen.fill((30, 30, 30))
        self._draw_grid()
        self._draw_status()
        pygame.display.flip()
        self.clock.tick(self.fps)

    def run(self):
        from entities import Action

        running = True
        paused = not self.auto_step

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        self.world._init_world()
                    elif not self.auto_step or paused:
                        action = _key_to_action(event.key)
                        if action is not None:
                            self.step(action)

            if self.auto_step and not paused:
                self.step(np.random.default_rng().choice([0, 1, 2, 3, 4, 5]))

            self.render()

        pygame.quit()


def _key_to_action(key):
    from entities import Action
    if key == pygame.K_UP:
        return Action.UP
    if key == pygame.K_DOWN:
        return Action.DOWN
    if key == pygame.K_LEFT:
        return Action.LEFT
    if key == pygame.K_RIGHT:
        return Action.RIGHT
    if key == pygame.K_e:
        return Action.EAT
    if key == pygame.K_SPACE:
        return Action.WAIT
    return None
