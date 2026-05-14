"""LTC Network — full decision network for the agent.

Takes observation + drive vector → hidden state evolution → action logits.
Uses LTCCell for continuous-time internal dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ltc_cell import LTCCell


class LTCNetwork(nn.Module):
    """LTC-based policy network for an Evola agent.

    Input:  flattened 5x5 obs (25) + drive vector (3) + optional self_vector (8) = 36 dims
    Output: action logits (6) + value estimate (1)
    """

    def __init__(self, input_size: int = 28, hidden_size: int = 64):
        super().__init__()
        self.input_size = input_size

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
        )
        self.ltc = LTCCell(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, 6)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, obs_flat: torch.Tensor, drive: torch.Tensor,
                h: torch.Tensor, dt: float = 0.1,
                self_vec: torch.Tensor | None = None,
                mem: torch.Tensor | None = None):
        x = torch.cat([obs_flat, drive], dim=-1)
        if self_vec is not None:
            x = torch.cat([x, self_vec], dim=-1)
        if mem is not None:
            x = torch.cat([x, mem], dim=-1)
        e = self.encoder(x)
        h_new = self.ltc(e, h, dt)
        logits = self.actor(h_new)
        value = self.critic(h_new)
        return logits, value, h_new

    def reset_hidden(self, batch_size: int = 1):
        return torch.zeros(batch_size, self.ltc.hidden_size)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
