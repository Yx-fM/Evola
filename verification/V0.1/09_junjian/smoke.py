"""Quick smoke test for EventBus + JunJian integration."""
import sys, os, tempfile, json

_BASE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_BASE, "..", "08_kunyu", "impl"))
sys.path.insert(0, os.path.join(_BASE, "..", "09_junjian", "impl"))

from event_bus import EventBus
from logger import JunJian

bus = EventBus()

with tempfile.TemporaryDirectory() as tmpdir:
    jj = JunJian(bus, log_dir=tmpdir).start()

    bus.publish("step.begin", {"step": 1, "agent_pos": (5, 3), "action": "UP"})
    bus.publish("step.outcome", {"step": 1, "energy_gained": 0.3, "damage_taken": 0.0})
    bus.publish("step.begin", {"step": 2, "agent_pos": (4, 3), "action": "EAT"})
    bus.publish("step.outcome", {"step": 2, "energy_gained": 0.3, "damage_taken": 0.0})

    jj.stop()

    entries = jj.get_log_entries()
    assert len(entries) == 4
    assert entries[0]["type"] == "step.begin"
    assert entries[3]["type"] == "step.outcome"
    print(f"OK: {len(entries)} events logged to {os.path.basename(jj.get_log_path())}")
