"""
Encoder networks for grid world observations.

Encoders take environment observations and map them to fixed-dimensional
embeddings that can be used by successor feature networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GridEncoder(nn.Module):
    """
    CNN-based encoder for grid world observations.

    Takes a grid observation (H, W, C) and encodes it to a fixed-dimensional
    vector. This is based on the encoder architecture from the simple_successor_features
    repo, adapted for Minigrid environments.
    """

    def __init__(self, obs_shape, feature_dim=128):
        """
        Args:
            obs_shape: Tuple (C, H, W) for input observation shape
            feature_dim: Output embedding dimension
        """
        super().__init__()

        assert len(obs_shape) == 3, "obs_shape should be (C, H, W)"

        self.obs_shape = obs_shape
        self.feature_dim = feature_dim

        # Simple CNN encoder
        # For minigrid, obs is typically (3, H, W) or similar
        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Calculate conv output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            conv_output = self.convnet(dummy_input)
            self.conv_output_size = conv_output.view(1, -1).size(1)

        # MLP to compress to feature_dim
        self.fc = nn.Sequential(
            nn.Linear(self.conv_output_size, feature_dim),
            nn.ReLU(),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, obs):
        """
        Args:
            obs: (batch, C, H, W) grid observation
        Returns:
            embedding: (batch, feature_dim) encoded representation
        """
        # Normalize observation if needed (minigrid observations are typically 0-255)
        if obs.dtype == torch.uint8 or obs.max() > 1.0:
            obs = obs.float() / 255.0

        h = self.convnet(obs)
        h = h.view(h.size(0), -1)
        embedding = self.fc(h)

        return embedding


class CoordinateEncoder(nn.Module):
    """
    Simple MLP encoder that takes (x, y) coordinates.

    This is the legacy encoder used in test_baseline.py. Kept for compatibility
    and comparison purposes.
    """

    def __init__(self, feature_dim=128, hidden_dim=128):
        """
        Args:
            feature_dim: Output embedding dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.feature_dim = feature_dim

        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, coords):
        """
        Args:
            coords: (batch, 2) normalized (x,y) coordinates in [0, 1]
        Returns:
            embedding: (batch, feature_dim) encoded representation
        """
        return self.net(coords)
