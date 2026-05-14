"""Quick test for Evola CLI with rich dashboard."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Monkey-patch to run non-interactively
import evola_cli
app = evola_cli.EvolaApp()

# Simulate a session
app.cmd_start()
assert app.platform_running
assert app.world is not None
print("OK start")

app.cmd_spawn("test_agent")
assert "test_agent" in app.agents
print("OK spawn")

app.cmd_run(50)
assert app.world.step_count == 50
print(f"OK run 50 steps")

app.cmd_extract("test_agent")
assert "test_agent" not in app.agents
assert os.path.exists(os.path.join(app.mind_dir, "test_agent.mind.evola"))
print("OK extract")

app.cmd_load("test_agent")
assert "test_agent" in app.agents
print("OK load")

app.cmd_extract("test_agent")
app.cmd_stop()
assert not app.platform_running
print("OK stop")

print("\nALL OK")
