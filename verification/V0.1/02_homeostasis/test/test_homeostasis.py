import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "impl"))

import pytest
from comfort_zones import ComfortZone, ENERGY_ZONE, NOVELTY_ZONE, SAFETY_ZONE
from variables import EnergyVariable, NoveltyVariable, SafetyVariable
from drives import Homeostasis, DriveVector


class TestComfortZone:
    def test_inside_zone(self):
        z = ComfortZone(0.5, 1.0)
        assert z.deviation(0.7) == 0.0
        assert z.is_comfortable(0.7)

    def test_below_zone(self):
        z = ComfortZone(0.5, 1.0)
        assert z.deviation(0.3) == 0.2
        assert not z.is_comfortable(0.3)

    def test_above_zone(self):
        z = ComfortZone(0.3, 0.8)
        assert z.deviation(0.95) == pytest.approx(0.15)
        assert not z.is_comfortable(0.95)

    def test_at_boundary(self):
        z = ComfortZone(0.5, 1.0)
        assert z.deviation(0.5) == 0.0
        assert z.deviation(1.0) == 0.0


class TestVariables:
    def test_energy_decays(self):
        v = EnergyVariable(initial=1.0, decay_rate=0.01)
        v.step()
        assert v.value < 1.0

    def test_energy_replenish(self):
        v = EnergyVariable(initial=0.5, decay_rate=0.0)
        v.replenish(0.3)
        assert v.value == 0.8

    def test_energy_clamped(self):
        v = EnergyVariable(initial=0.5, decay_rate=0.0)
        v.replenish(2.0)
        assert v.value == 1.0
        v.deplete(2.0)
        assert v.value == 0.0

    def test_novelty_starts_high(self):
        v = NoveltyVariable(total_cells=100)
        assert v.value > 0.9  # nothing visited yet

    def test_novelty_decreases_with_visits(self):
        v = NoveltyVariable(total_cells=100)
        for _ in range(50):
            v.visit()
        assert v.value < 0.6

    def test_novelty_saturates(self):
        v = NoveltyVariable(total_cells=10)
        for _ in range(100):
            v.visit()
        assert v.value == 0.0  # all cells visited

    def test_safety_rises_with_danger(self):
        v = SafetyVariable()
        for _ in range(5):
            v.step(True)
        assert v.value > 0.5

    def test_safety_decays_without_danger(self):
        v = SafetyVariable()
        v.step(True)
        v.step(True)
        for _ in range(40):
            v.step(False)
        assert v.value < 0.3


class TestHomeostasis:
    def test_initial_drive_on_border(self):
        h = Homeostasis(total_cells=100, energy_decay=0.0)
        drive = h.get_drive()
        # Starts at 0.8 energy, which is in comfort [0.5, 1.0]
        assert drive.energy == 0.0

    def test_energy_drive_rises_when_hungry(self):
        h = Homeostasis(total_cells=100, energy_decay=0.0)
        h.energy.value = 0.3  # manually set below comfort
        drive = h.get_drive()
        assert drive.energy > 0.0

    def test_eating_reduces_energy_drive(self):
        h = Homeostasis(total_cells=100, energy_decay=0.01)
        for _ in range(50):  # decay energy
            h.update({"agent_pos": (0, 0)}, dt=1.0)
        drive_before = h.get_drive()
        h.update({"energy_gained": 0.5, "agent_pos": (0, 0)}, dt=1.0)
        drive_after = h.get_drive()
        assert drive_after.energy < drive_before.energy

    def test_visit_tracks_novelty(self):
        h = Homeostasis(total_cells=100)
        for i in range(20):
            h.update({"agent_pos": (i, 0)}, dt=1.0)
        assert h.novelty.visited_count == 20
        assert h.novelty.value < 0.9

    def test_danger_updates_safety_drive(self):
        h = Homeostasis(total_cells=100)
        drive_before = h.get_drive()
        assert drive_before.safety == 0.0
        h.update({"danger_visible": True, "agent_pos": (0, 0)}, dt=1.0)
        drive_after = h.get_drive()
        assert drive_after.safety > 0.0

    def test_drive_labels(self):
        h = Homeostasis(total_cells=100)
        h.energy.value = 0.3
        h.safety.value = 0.7
        labels = h.get_state()["labels"]
        assert labels["energy"] in ("starving", "hungry")
        assert labels["safety"] in ("alert", "scared")

    def test_get_state(self):
        h = Homeostasis(total_cells=50)
        state = h.get_state()
        assert "energy" in state
        assert "novelty" in state
        assert "safety" in state
        assert "drive" in state
        assert "labels" in state
