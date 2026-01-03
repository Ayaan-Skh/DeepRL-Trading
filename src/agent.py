import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DQNAgent:
    """
    Deep Q-Network Agent for trading
    
    Compponents:
    - Main Network(Q_main): Updated every step
    - Target network(Q_target): Updated every N steps
    - Replay Buffer: Stores experinces
    - Epsilon Greedy: Exploration tendency    
    
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=0.001,
        gamma=1,
        epsilon_start=1,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=32,
        target_update_freq=1000
        ):
        """
        Docstring for __init__
         
            state_dim: Size of the vector (13 for us) 
            action_dim: Number of actions(3:HOLD, SELL, BUY)
            learning_rate: Adam optimizer learning rate 
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay per episode
            buffer_capacity: Max experinces to store
            batch_size: Mini- batch size for training
            target_update_freq: steps between target network updates
        """

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps_done = 0
        
        # Initialize networks
        from src.model import DQN
        
        self.main_network=DQN(state_dim,action_dim)        
        self.target_network=DQN(state_dim,action_dim)
        
        # Import weights from main to target network
        self.target_network.load_state_dict(self.main_network.state_dict())
        # .eval() puts target into evaluation mode so that it dosent trains
        self.target_network.eval()
        
        # Initialize Optimizers
        self.optimizer=optim.Adam(
            self.main_network.parameters(),
            lr=learning_rate
            #It updates main network weights
        )
        
        # Initialize replay buffer
        self.replay_buffer=deque(maxlen=buffer_capacity)
        
    # Step 2: Action selection    
    def select_action(self,state,evaluation=False):
        """
        Select an strategy using epsilon greedy strategy
        
        Args:
            state: Current state(numpy array)
            evaluation: If true, no exploration only exploit
        
        Returns:
            action: Integer(0,1,2)
        """
        if evaluation:
            epsilon=0.0 # For testing
        else:
            epsilon=self.epsilon # For training
        
        if random.random()<epsilon:
            return random.randint(0,self.action_dim-1)
        else:
            with torch.no_grad():# Dosent compute gradients
                state_tensor=torch.FloatTensor(state).unsqueeze(0)# This converts the input in a tensor of typr float 32 bits that can be given as input to the 
                
                """
                  What this does:

                    Get Q-values: [Q(HOLD), Q(BUY), Q(SELL)]
                    Find index of max: argmax([0.5, 2.3, -0.8]) = 1
                    Convert to Python int: .item()
                """
                q_values=self.main_network(state_tensor)
                action=q_values.argmax(dim=1).item()                    
                    
            return action
        
    # STEP 3: Storing Experinces
    def remember(self,state,action,reward,next_state,done):
        """
         Store experience in replay buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Is episode finished ??
        """    
        self.replay_buffer.append(state,action,reward,next_state,done)
    
    # STEP 4: Training
    def train(self):
        """
        Sample batch from replay buffer and tran the main network
        
        Returns:
            loss: Training loss, or None if not enough data
        """    
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample randome batch
        batch=random.sample(self.replay_buffer,self.batch_size)
        
        states,actions,rewards,next_states,dones=zip(*batch)
        
        states=torch.FloatTensor(np.array(states))
        actions=torch.LongTensor(actions)
        rewards=torch.FloatTensor(rewards)
        next_states=torch.FloatTensor(np.array(next_states))
        dones=torch.FloatTensor(dones)
        
        # Compute current Q-values
        current_q_values = self.main_network(states)  # Shape: (32, 3)

        # .unsqueeze() converts [batch_size] to [batch_size,1]
        # .squeeze() converts [batch_size,1] to [batch_size]
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze()
        
        #Compute target Q values(for target network)    
        with torch.no_grad(): # DOnt ocmpute gradients for target
            next_q_values=self.target_network(next_states) #(32,3)
            max_next_q=next_q_values.max(dim=1)[0] #(32,)
            
            # Target = reward + gamma * max_next_q (if not done)
            target_q=rewards+self.gamma*max_next_q*(1-dones)
        
        # Compute Loss
        loss=nn.MSELoss()(current_q,target_q)    
        
        # Backpropogation
        self.optimizer.zero_grad() # Clear old gradients
        loss.backward() # Compute new gradients
        
        # Gradient clipping (preventing exploding gradients)
        torch.utils.clip_grad_norm_(self.main_network.parameters(),max_norm=10)
        self.optimizer.step()  # Update weights
        # zero_grad(): Clear previous gradients (don't accumulate)
        # backward(): Compute ∂loss/∂weight for all weights
        # clip_grad_norm_(): Cap gradients at 10 (prevent explosion)
        # step(): Update weights using gradients
        
        self.steps_done+=1
        if self.steps_done % self.target_update_freq==0:
            self.update_target_network()
        
        return loss.item()
    
    
    def update_target_network(self):
        """
        Copy weights from target to mai network
        """
        self.target_network.load_state_dict(self.main_network.state_dict())
        print(f"Target network updated at step: {self.steps_done}")
        
    def decay_eplsilon(self):
        self.epsilon=max(self.epsilon_end,self.epsilon*self.epsilon_decay)
    
    
    def save(self, filepath):
        """Save agent state to file"""
        torch.save({
            'main_network': self.main_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, filepath)
        print(f"Agent saved to {filepath}")
        
        
        
    def load(self, filepath):
        """Load agent state from file"""
        checkpoint = torch.load(filepath)
        self.main_network.load_state_dict(checkpoint['main_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        print(f"Agent loaded from {filepath}")    
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        