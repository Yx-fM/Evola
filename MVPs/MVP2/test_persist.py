"""Test round-trip persistence."""
import sys, os, tempfile, json

_BASE = os.path.dirname(__file__)
_PROJ = os.path.dirname(os.path.dirname(_BASE))
sys.path.insert(0, os.path.join(_PROJ, "verification", "V0.1", "10_sensorimotor", "impl"))
sys.path.insert(0, os.path.join(_PROJ, "verification", "V0.1", "08_kunyu", "impl"))
sys.path.insert(0, os.path.join(_PROJ, "verification", "V0.1", "02_homeostasis", "impl"))
sys.path.insert(0, os.path.join(_PROJ, "verification", "V0.1", "01_liquid_dynamics", "impl"))

import numpy as np
from event_bus import EventBus
from world import KunyuWorld, WorldConfig
from agent import SensorimotorAgent
from mvp_2_config import WORLD, HOMEOSTASIS, ARBITER

world = KunyuWorld(WorldConfig(**WORLD))
agent = SensorimotorAgent(agent_id="test", arbiter_type="ltc",
                           ltc_hidden=ARBITER["ltc_hidden"], ltc_lr=ARBITER["ltc_lr"])
agent.set_total_cells(300)
world.register_agent("test")

# Run some steps to build state
for _ in range(100):
    obs = world.get_obs("test")
    action = int(agent.act(obs))
    _, info = world.step(action, agent_id="test")
    agent.observe(info)

# Save
with tempfile.NamedTemporaryFile(suffix=".mind.evola", delete=False) as f:
    mind_path = f.name
agent.save_mind(mind_path)
file_size = os.path.getsize(mind_path)
print(f"Saved: {file_size} bytes")

# Verify file is valid JSON and has weights
with open(mind_path) as f:
    data = json.load(f)
print(f"Format: {data['format']} v{data['version']}")
print(f"Steps: {data['step_count']}")
print(f"Has ltc_weights: {'ltc_weights' in data} ({len(data.get('ltc_weights',''))} chars)")
print(f"Has memory: {'memory' in data}")
if 'memory' in data:
    print(f"  STM: {len(data['memory']['stm'])} LTM: {len(data['memory']['ltm'])}")
print(f"Has self_model: {'self_model' in data}")

# Load into new agent
agent2 = SensorimotorAgent.load_mind(mind_path, arbiter_type="ltc",
                                       arbiter_params=ARBITER)
agent2.set_total_cells(300)

# Verify state
assert agent2.step_count == agent.step_count
assert len(agent2.visited) == len(agent.visited)
assert abs(agent2.homeo.energy.value - agent.homeo.energy.value) < 0.01
print(f"\nRound-trip OK: steps={agent2.step_count} energy={agent2.homeo.energy.value:.2f} visited={len(agent2.visited)}")

# Verify LTC can still act
world.register_agent("test2")
obs = world.get_obs("test2")
action = agent2.act(obs)
print(f"Acts OK: action={action}")

os.unlink(mind_path)
print("\nALL GOOD")

