import torch
import torch.nn as nn

# the actual architecture of the Physics-Guided Neural Network

class PhysicsGuidedNN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super(PhysicsGuidedNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(), # Differentiable activation for autograd
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        return self.network(x)