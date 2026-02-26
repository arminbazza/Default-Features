"""
Repair networks for learning corrections to baseline representations.

These networks learn how to adapt a frozen baseline representation (DR or SF)
when the environment dynamics change (e.g., barrier added).
"""

import torch.nn as nn


class RepairNetwork(nn.Module):
    """
    Learns corrections Δψ(s) to a frozen baseline SF representation.

    New architecture that operates in SF latent space:
    - Input: state embedding (feature_dim)
    - Output: ΔSF correction vector (sf_dim)
    - Combined: ψ_new(s) = ψ_baseline(s) + Δψ(s)

    This allows the network to scale independently of environment size.
    """

    def __init__(self, feature_dim, sf_dim, hidden_dim=128, rank=None):
        """
        Args:
            feature_dim: Dimensionality of input state embedding
            sf_dim: Dimensionality of SF space (output correction dimension)
            hidden_dim: Hidden layer dimension
            rank: If provided, use low-rank bottleneck (LoRA-style)
        """
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
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)  # Small init for corrections
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, embedding):
        """
        Args:
            embedding: (batch, feature_dim) state embedding from encoder
        Returns:
            delta_sf: (batch, sf_dim) correction to SF vector
        """
        if self.rank is not None:
            h = self.encoder(embedding)
            delta_sf = self.decoder(h)
        else:
            delta_sf = self.net(embedding)

        return delta_sf


class TestRepairNetwork(nn.Module):
    """
    [LEGACY - for test_baseline.py]

    Learns the correction Δψ(s) to the frozen baseline DR.

    Full prediction: ψ_new(s) = D_os[s, :] + Δψ(s)

    Two modes:
        - Full: MLP maps (x,y) → R^num_states
        - Low-rank: MLP maps (x,y) → R^rank, then linear projection to R^num_states
          (This is the LoRA-style bottleneck)

    This version operates on coordinate inputs and outputs full state×state
    corrections, which doesn't scale well to larger environments.
    """

    def __init__(self, num_states, hidden_dim=128, rank=None):
        """
        Args:
            num_states: size of state space (e.g., 400 for 20x20)
            hidden_dim: width of hidden layers
            rank: if provided, use low-rank bottleneck of this dimension.
                  if None, use full-rank output.
        """
        super().__init__()
        self.num_states = num_states
        self.rank = rank

        if rank is not None:
            self.encoder = nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, rank),
            )
            # No bias — the correction should be zero-centered
            self.decoder = nn.Linear(rank, num_states, bias=False)
        else:
            self.net = nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_states),
            )

    def forward(self, coords):
        """
        Args:
            coords: (batch, 2) normalized (x,y) in [0, 1]
        Returns:
            delta_psi: (batch, num_states) correction to DR row
        """
        if self.rank is not None:
            h = self.encoder(coords)
            return self.decoder(h)
        else:
            return self.net(coords)
