import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DQN,self).__init__()
        
        self.network=nn.Sequential(
            nn.Linear(state_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2,action_dim)
        )
        
        def forward(self,state):
            return self.network(state)