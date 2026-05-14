import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from evola_cli import EvolaApp
app = EvolaApp()
app.cmd_start("tiny")
app.cmd_spawn("test")
app.cmd_run(100)
h = app.agents["test"].homeo
print(f"E={h.energy.value:.2f} N={h.novelty.value:.2f} S={h.safety.value:.2f}")
print(f"Steps={app.world.step_count} Visited={len(app.agents['test'].visited)}")
print("OK")
