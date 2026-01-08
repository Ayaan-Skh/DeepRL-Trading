import torch
import torch.nn as nn

class DQN(nn.Module):
    """
        Deep Q-Network that estimates Q values
        Architecture:
            Input(13)-> [64] -> ReLU -> [64] -> ReLU -> [32] -> ReLU -> Output(3)

        Input:
            State Vector (13 features)
            
        Output:
            Q-values for each action[Q(Hold), Q(Buy), Q(Sell)]    
    
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DQN,self).__init__()
        
         # Define network layers
        self.network=nn.Sequential(
            # Layer 1: Input -> Hidden
            nn.Linear(state_dim,hidden_dim),
            nn.ReLU(),
            
            # Layer 2: Hidden -> Hidden
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            
            # Layer 3: Hidden -> Smaller Hidden
            nn.Linear(hidden_dim,hidden_dim//2),
            nn.ReLU(),
            
            # Layer 4: Hidden -> Output
            nn.Linear(hidden_dim//2,action_dim)
            # No activation on output! (Q-values can be negative)
        )
        
        # Initialize weights using Xavier Initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        """
        Initialize network weights using Xavier uniform initialization.
        """    
        for module in self.modules():
            if isinstance(module,nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant(module.bias,0.0)
                # For each Linear layer:

                # Weights: Xavier uniform (good for ReLU)
                # Biases: All zeros
                
                
    def forward(self,state):
        """
        Forward pass: State -> Q-values
        
        Args:
            state: Tensor of shape (batch_size,state_dim) or (state_dim)
        
        Returns:
            q_values: Tensor of shale (batch_size, action_dim) or (action_dim)    
        """
        return self.network(state)
        # Simple! Just pass state through the sequential network.
    
    
                
        
        