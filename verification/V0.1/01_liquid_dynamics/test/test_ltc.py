"""Tests for LTC cell, network, and arbiter."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "impl"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "10_sensorimotor", "impl"))

import pytest
import numpy as np
import torch
from ltc_cell import LTCCell
from ltc_network import LTCNetwork
from ltc_arbiter import LTCArbiter
from behavior_standard.arbitration import DriveVector, Facts


class TestLTCCell:
    def test_creation(self):
        cell = LTCCell(8, 32)
        assert cell.hidden_size == 32
        assert cell.log_tau.shape == (32,)

    def test_tau_positive(self):
        cell = LTCCell(8, 16)
        assert (cell.tau > 0).all()

    def test_forward_shape(self):
        cell = LTCCell(4, 8)
        x = torch.randn(2, 4)
        h = torch.zeros(2, 8)
        h2 = cell(x, h, dt=0.1)
        assert h2.shape == (2, 8)
        assert not torch.allclose(h2, h)  # should change

    def test_hidden_evolves(self):
        cell = LTCCell(4, 8)
        x = torch.randn(1, 4)
        h = torch.zeros(1, 8)
        for _ in range(10):
            h = cell(x, h, dt=0.1)
        assert h.abs().sum() > 0.01  # accumulated activity

    def test_small_dt_conservative(self):
        """Small dt: less change per step."""
        cell = LTCCell(4, 8)
        x = torch.randn(1, 4)
        h0 = torch.zeros(1, 8)
        h1 = cell(x, h0.clone(), dt=0.01)
        h2 = cell(x, h0.clone(), dt=0.5)
        diff_small = (h1 - h0).abs().sum()
        diff_large = (h2 - h0).abs().sum()
        assert diff_small < diff_large


class TestLTCNetwork:
    def test_creation(self):
        net = LTCNetwork(input_size=28, hidden_size=32)
        assert net.total_params() > 0

    def test_forward_shape(self):
        net = LTCNetwork(input_size=28, hidden_size=32)
        obs = torch.randn(2, 25)
        drive = torch.randn(2, 3)
        h = net.reset_hidden(2)
        logits, value, h2 = net(obs, drive, h)
        assert logits.shape == (2, 6)
        assert value.shape == (2, 1)
        assert h2.shape == (2, 32)

    def test_hidden_persistence(self):
        """Hidden state carries information across steps."""
        net = LTCNetwork(input_size=28, hidden_size=32)
        obs = torch.randn(1, 25)
        drive = torch.randn(1, 3)
        h = net.reset_hidden(1)
        # Step 1
        logits1, _, h = net(obs, drive, h)
        # Step 2: same input, different hidden → different output
        logits2, _, h = net(obs, drive, h)
        assert not torch.allclose(logits1, logits2)

    def test_params_small(self):
        """Network should be lightweight (<5000 params for small hidden)."""
        net = LTCNetwork(input_size=28, hidden_size=16)
        p = net.total_params()
        assert p < 5000, f"Too many params: {p}"


class TestLTCArbiter:
    def test_creation(self):
        arb = LTCArbiter(hidden_size=16, lr=0.01)
        assert arb.net is not None
        assert arb.h is None

    def test_select_returns_primitive(self):
        arb = LTCArbiter(hidden_size=16)
        drive = DriveVector(energy=0.5, novelty=0.3, safety=0.0)
        facts = Facts(food_dirs=["UP"], danger_dirs=[])
        from behavior_standard.arbitration import SalienceMap, DecisionContext
        from behavior_standard.primitives import BehaviorPrimitive
        p = arb.select(drive, facts, SalienceMap(), DecisionContext())
        assert isinstance(p, BehaviorPrimitive)

    def test_update_trains(self):
        """One training step should change actor-side parameters."""
        arb = LTCArbiter(hidden_size=16, lr=0.1)
        skip = {"log_tau", "critic"}
        before = {n: p.clone() for n, p in arb.net.named_parameters()
                  if not any(s in n for s in skip)}

        drive = DriveVector(energy=0.8, novelty=0.3, safety=0.0)
        facts = Facts(food_dirs=["UP"])
        from behavior_standard.arbitration import SalienceMap, DecisionContext
        arb.select(drive, facts, SalienceMap(), DecisionContext())

        drive_after = DriveVector(energy=0.3, novelty=0.2, safety=0.0)
        arb.update(drive_after)

        changed = 0
        for n, p in arb.net.named_parameters():
            if any(s in n for s in skip):
                continue
            if not torch.allclose(before[n], p):
                changed += 1
        assert changed > 0, "No actor-side params changed"

    def test_hidden_persists_across_selects(self):
        arb = LTCArbiter(hidden_size=16)
        drive = DriveVector(energy=0.5, novelty=0.3, safety=0.0)
        facts = Facts()
        from behavior_standard.arbitration import SalienceMap, DecisionContext
        p1 = arb.select(drive, facts, SalienceMap(), DecisionContext())
        p2 = arb.select(drive, facts, SalienceMap(), DecisionContext())
        # Different hidden states can lead to different actions
        # (not guaranteed, but the hidden IS persisting)
        assert arb.h is not None
