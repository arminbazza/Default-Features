"""
Successor Feature (SF) networks.

These networks take encoded state representations and predict successor features
in a fixed-dimensional latent space (rather than full state×state matrices).
"""

import torch.nn as nn
import torch.nn.functional as F


class SFNetwork(nn.Module):
    """
    Successor Feature network that predicts SF vectors from state embeddings.

    This network maps from a state embedding to a successor feature vector
    in the same embedding space. The key insight is that both the state
    features φ(s) and the successor features ψ(s) live in the same latent space.

    Architecture inspired by simple_successor_features (Chua et al., NeurIPS 2024).
    """

    def __init__(self, feature_dim, sf_dim, hidden_dim=128):
        """
        Args:
            feature_dim: Dimensionality of input state embedding
            sf_dim: Dimensionality of successor feature output
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.sf_dim = sf_dim
        self.hidden_dim = hidden_dim

        # MLP to predict successor features
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, sf_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, embedding, normalize=True):
        """
        Args:
            embedding: (batch, feature_dim) state embedding from encoder
            normalize: If True, L2-normalize the output SF vector
        Returns:
            sf: (batch, sf_dim) successor feature vector
        """
        sf = self.net(embedding)

        if normalize:
            sf = F.normalize(sf, p=2, dim=-1)

        return sf


class BasisFeaturesNetwork(nn.Module):
    """
    Network that predicts basis features φ(s) from state embeddings.

    In the successor features framework:
    - φ(s): basis features (what we observe/care about in state s)
    - ψ(s): successor features (expected discounted future φ)

    This network predicts φ(s) from the encoded state representation.
    """

    def __init__(self, feature_dim, sf_dim, hidden_dim=128):
        """
        Args:
            feature_dim: Dimensionality of input state embedding
            sf_dim: Dimensionality of basis feature output
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.sf_dim = sf_dim
        self.hidden_dim = hidden_dim

        # MLP to predict basis features
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, sf_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, embedding, normalize=True):
        """
        Args:
            embedding: (batch, feature_dim) state embedding from encoder
            normalize: If True, L2-normalize the output basis features
        Returns:
            basis_features: (batch, sf_dim) basis feature vector
        """
        phi = self.net(embedding)

        if normalize:
            phi = F.normalize(phi, p=2, dim=-1)

        return phi
