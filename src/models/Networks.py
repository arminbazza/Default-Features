import torch
import torch.nn as nn


class RepairNetwork(nn.Module):
    """
    Learns the correction Δψ(s) to the frozen baseline DR.
    
    Full prediction: ψ_new(s) = D_os[s, :] + Δψ(s)
    
    Two modes:
        - Full: MLP maps (x,y) → R^num_states
        - Low-rank: MLP maps (x,y) → R^rank, then linear projection to R^num_states
          (This is the LoRA-style bottleneck)
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