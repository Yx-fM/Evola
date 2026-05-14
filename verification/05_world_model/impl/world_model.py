"""World Model — predicts environment transitions for mental simulation.

Small MLP: (obs_flat(25) + action_onehot(6)) → pred_delta_obs(25).
Trained online with MSE on the residual. Used for 1-step planning:
for each candidate action, predict the result, estimate drive change, bias action.

This gives the agent a primitive "imagination" — "if I go UP, what happens?"
"""

import torch
import torch.nn as nn


class WorldModel(nn.Module):
    def __init__(self, input_size: int = 31, hidden_size: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 25),  # predict delta observation
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, obs_flat, action_idx):
        """obs_flat: [batch, 25], action_idx: [batch]"""
        action_onehot = torch.zeros(len(action_idx), 6, device=obs_flat.device)
        action_onehot.scatter_(1, action_idx.unsqueeze(1), 1.0)
        x = torch.cat([obs_flat, action_onehot], dim=-1)
        return self.net(x)

    def predict(self, obs_flat, action_idx):
        """Predict delta_obs for a single action."""
        with torch.no_grad():
            return self.forward(obs_flat.unsqueeze(0),
                                torch.tensor([action_idx], device=obs_flat.device)).squeeze(0)

    def train_step(self, obs, action, next_obs, optimizer):
        """One supervised training step. Returns loss."""
        delta_true = next_obs - obs
        delta_pred = self.forward(obs.unsqueeze(0),
                                   torch.tensor([action], device=obs.device))
        loss = self.loss_fn(delta_pred, delta_true.unsqueeze(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def estimate_drive_delta(self, obs_flat, action_idx, drive_fn):
        """Heuristic: predict next obs and estimate drive change.
        drive_fn(obs_flat) → (energy_drive, novelty_drive, safety_drive)
        """
        next_obs = obs_flat + self.predict(obs_flat, action_idx)
        current_drive = sum(drive_fn(obs_flat))
        next_drive = sum(drive_fn(next_obs))
        return current_drive - next_drive  # positive = improvement
