"""LTC Cell — Liquid Time-Constant neuron layer.

Core equation (continuous-time):
    dx/dt = -(1/τ) ⊙ x + W·f(x + I)

Where τ is a per-neuron learnable time constant (> 0), and f = tanh.

Uses simple Euler integration to avoid torchdiffeq dependency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LTCCell(nn.Module):
    """A single layer of LTC neurons with per-neuron time constants.

    Args:
        input_size:   dimensionality of input I(t)
        hidden_size:  number of LTC neurons
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Synaptic weights: map input+hidden → pre-activation
        self.W_in = nn.Linear(input_size + hidden_size, hidden_size, bias=True)

        # Per-neuron time-constant parameter (log-space for positivity)
        self.log_tau = nn.Parameter(torch.zeros(hidden_size))

    @property
    def tau(self):
        return F.softplus(self.log_tau) + 0.01

    def forward(self, x: torch.Tensor, h: torch.Tensor, dt: float = 0.1):
        """One integration step. h(t+dt) ← h(t) + dt * dh/dt.

        Args:
            x:  input at current time [batch, input_size]
            h:  hidden state [batch, hidden_size]
            dt: integration step size

        Returns:
            h_new: updated hidden state
        """
        # dh/dt = -(1/τ) ⊙ h + W·tanh(x || h)
        combined = torch.cat([x, h], dim=-1)
        syn = torch.tanh(self.W_in(combined))
        dh = syn - h / self.tau
        return h + dt * dh
