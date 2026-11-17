import torch
import torch.nn as nn
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform


class BoundedNormal(TransformedDistribution):
    """Keep stored float32 actions away from tanh's singular endpoints."""

    def sample(self, sample_shape=torch.Size()):
        return super().sample(sample_shape).clamp(-1.0 + 1e-6, 1.0 - 1e-6)

    def log_prob(self, value):
        return super().log_prob(value.clamp(-1.0 + 1e-6, 1.0 - 1e-6))


#actor network - outputs mean and std of the action distribution
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, alpha, fc1_dims=256, fc2_dims=256):
        super(ActorNetwork, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU()
        )
        
        #outputs mean of the action distribution for each action dimension 
        self.mu = nn.Linear(fc2_dims, action_dim)

        #creates learnable parameter that can be updated during training using log of the standard deviation to ensure it stays positive 
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        
    def forward(self, state):
        x = self.actor(state)
        mu = self.mu(x)
        std = torch.exp(self.log_std.clamp(-5.0, 2.0))
        
        dist = BoundedNormal(
            Normal(mu, std),
            [TanhTransform(cache_size=1)],
        )
        return dist


#critic network - outputs single value (expected total future reward) of the state
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, alpha, fc1_dims=256, fc2_dims=256):
        super(CriticNetwork, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        
    def forward(self, state):
        value = self.critic(state)
        return value
