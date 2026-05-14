import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "impl"))

import torch
import numpy as np
from world_model import WorldModel


def test_creation():
    wm = WorldModel()
    assert wm is not None

def test_forward_shape():
    wm = WorldModel()
    obs = torch.randn(2, 25)
    actions = torch.tensor([0, 3])
    out = wm(obs, actions)
    assert out.shape == (2, 25)

def test_predict_shape():
    wm = WorldModel()
    obs = torch.randn(25)
    delta = wm.predict(obs, 2)
    assert delta.shape == (25,)

def test_training_reduces_loss():
    wm = WorldModel(hidden_size=16)
    opt = torch.optim.Adam(wm.parameters(), lr=0.01)
    obs = torch.randn(25)
    next_obs = obs + torch.randn(25) * 0.1
    action = 2
    losses = []
    for _ in range(50):
        loss = wm.train_step(obs, action, next_obs, opt)
        losses.append(loss)
    assert losses[-1] < losses[0] * 0.8  # loss decreased

def test_estimate_drive_delta():
    wm = WorldModel(hidden_size=16)
    obs = torch.zeros(25)
    obs[1] = 1.0  # food UP
    obs[6] = 1.0  # danger at idx 6

    def fake_drive(o):
        return (float(o[1:5].sum()), float(o[6:10].sum()), 0.0)

    # Action 0 (UP) should reduce food drive
    delta = wm.estimate_drive_delta(obs, 0, fake_drive)
    assert isinstance(delta, float)
