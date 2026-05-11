"""Terminal renderer - ANSI escape codes, zero dependencies."""

import os
import sys


TO_CHAR = {
    0: " ",    # EMPTY
    1: "#",    # WALL
    2: "\033[32m*\033[0m",  # FOOD - green
    3: "\033[31m!\033[0m",  # DANGER - red
    4: "\033[34mA\033[0m",  # AGENT - blue
}


def render(world):
    state = world.get_state()
    grid = state["grid"]
    h, w = grid.shape
    agent_r, agent_c = state["agent_pos"]

    os.system("cls" if sys.platform == "win32" else "clear")

    lines = [
        f"=== Kunyu World ===  Step: {state['step']} | Agent: {(agent_r, agent_c)} | Visited: {state['n_visited']}",
        "+" + "-" * w + "+",
    ]
    for r in range(h):
        row = "|"
        for c in range(w):
            row += TO_CHAR.get(grid[r, c], "?")
        row += "|"
        lines.append(row)
    lines.append("+" + "-" * w + "+")
    print("\n".join(lines))


def render_frame(world, out_lines: list):
    """Append a compact ASCII frame for batch output."""
    grid = world.get_full_grid()
    h, w = grid.shape
    out_lines.append(f"--- Step {world.step_count} ---")
    for r in range(h):
        row = ""
        for c in range(w):
            cell = grid[r, c]
            if cell == 0:
                row += "."
            elif cell == 1:
                row += "#"
            elif cell == 2:
                row += "F"
            elif cell == 3:
                row += "X"
            elif cell == 4:
                row += "A"
        out_lines.append(row)
