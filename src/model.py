"""Model definition for the Physics-Guided Neural Network (PGNN).

This module keeps the architecture intentionally compact:
- fully connected layers for tabular PEMFC features,
- smooth activations (Tanh) so input derivatives exist for physics loss,
- scalar output for voltage prediction.
"""

import torch
import torch.nn as nn

class PhysicsGuidedNN(nn.Module):
    """Simple MLP used by the PGNN training/evaluation pipeline."""

    def __init__(self, input_dim=3, hidden_dim=64):
        """Initialize network layers.

        Args:
            input_dim: Number of input features.
            hidden_dim: Width of hidden fully connected layers.
        """
        super(PhysicsGuidedNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        """Predict scaled voltage from scaled input features."""
        return self.network(x)