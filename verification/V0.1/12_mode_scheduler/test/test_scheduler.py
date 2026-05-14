import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "impl"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "06_hot_warm_cold", "impl"))
from scheduler import ModeScheduler
from modes import AgentMode


def test_starts_hot():
    ms = ModeScheduler()
    assert ms.current == AgentMode.HOT

def test_danger_forces_hot():
    ms = ModeScheduler()
    ms.current = AgentMode.WARM
    m = ms.update(energy=0.8, novelty=0.3, safety=0.8)
    assert m == AgentMode.HOT

def test_energy_drop_forces_hot():
    ms = ModeScheduler()
    ms.current = AgentMode.WARM
    m = ms.update(energy=0.2, novelty=0.3, safety=0.0)
    assert m == AgentMode.HOT

def test_hot_to_warm_when_safe():
    ms = ModeScheduler()
    ms.current = AgentMode.HOT
    m = ms.update(energy=0.8, novelty=0.3, safety=0.0)
    assert m == AgentMode.WARM

def test_warm_hysteresis():
    ms = ModeScheduler()
    # Go to WARM
    ms.update(energy=0.8, novelty=0.3, safety=0.0)
    ms.update(energy=0.8, novelty=0.3, safety=0.0)
    assert ms.current == AgentMode.WARM
    # Small energy drop should NOT trigger HOT (hysteresis)
    m = ms.update(energy=0.4, novelty=0.3, safety=0.0)
    assert m == AgentMode.WARM  # still warm
    # Big drop should trigger
    m = ms.update(energy=0.2, novelty=0.3, safety=0.0)
    assert m == AgentMode.HOT
