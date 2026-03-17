"""
Repair networks for learning corrections to baseline representations.

These networks learn how to adapt a frozen baseline representation (DR or SF)
when the environment dynamics change (e.g., barrier added).
"""

import torch.nn as nn


class RepairNetworkSF(nn.Module):
    """
    Learns corrections Δψ(s) to a frozen baseline SF representation.
    """

    def __init__(self, feature_dim, sf_dim, hidden_dim=128, rank=None):
        super().__init__()

        self.feature_dim = feature_dim
        self.sf_dim = sf_dim
        self.hidden_dim = hidden_dim
        self.rank = rank

        if rank is not None:
            # Low-rank bottleneck (LoRA-style)
            self.encoder = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, rank),
            )
            # No bias — the correction should be zero-centered
            self.decoder = nn.Linear(rank, sf_dim, bias=False)
        else:
            # Full-rank correction
            self.net = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, sf_dim),
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)  # Small init for corrections
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, embedding):
        if self.rank is not None:
            h = self.encoder(embedding)
            delta_sf = self.decoder(h)
        else:
            delta_sf = self.net(embedding)

        return delta_sf


class RepairNetworkLinear(nn.Module):
    """
    Learns the correction Δψ(s) to the frozen baseline DR.
    """

    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.repair = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        y = self.repair(x)
        return y
