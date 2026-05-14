"""Save a fresh v0.1 Evola model (untrained baseline)."""
import sys, os
_PROJ = r"Q:\All_Items\DreamProjects\Evola"

sys.path.insert(0, os.path.join(_PROJ, "verification", "V0.1", "08_kunyu", "impl"))
sys.path.insert(0, os.path.join(_PROJ, "verification", "V0.1", "02_homeostasis", "impl"))
sys.path.insert(0, os.path.join(_PROJ, "verification", "V0.1", "10_sensorimotor", "impl"))

from world import KunyuWorld, WorldConfig
from event_bus import EventBus
from agent import SensorimotorAgent

WORLD = {"height": 15, "width": 20, "n_food": 8, "n_dangers": 2}

# Create world + fresh agent
bus = EventBus()
world = KunyuWorld(WorldConfig(**WORLD), bus=bus)
agent = SensorimotorAgent(
    agent_id="evola_v0.1_untrained",
    arbiter_type="ltc", ltc_hidden=64, ltc_lr=0.001, energy_decay=0.003,
    arbiter_params={"safety_threshold": 0.3, "energy_urgent": 0.5, "energy_safe": 0.3, "novelty_threshold": 0.2},
)
agent.set_total_cells(world.cfg.height * world.cfg.width)

# Save immediately (no training)
path = os.path.join(os.path.dirname(__file__), "v0.1", "evola_v0.1_untrained.mind.evola")
agent.save_mind(path)

import os as _os
size = _os.path.getsize(path)
print(f"Saved untrained model: {path} ({size} bytes)")
print(f"  Energy: {agent.homeo.energy.value:.2f}")
print(f"  Steps: {agent.step_count}")
print(f"  Visited: {len(agent.visited)}")
