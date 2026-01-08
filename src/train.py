import sys 
import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   
import matplotlib.pyplot as plt
import pathlib as Path

from src.agent import DQNAgent
from src.enviornments import TradingEnvironment
from src.utils import set_seed,create_directories

def train_agent(
    num_episodes=500,
    initial_balance=1000000,
    config=None
    ):
    """
    Train DQN agent on stock trading
    
    num_episodes: Number of episodes
    initial_balance: Starting Cash
    config: Optional config dict(Can override defaults)
    
    """
    set_seed(42)
    create_directories()
    
    print("="*60)
    print(f"Deep Q Trading Agent - Training")
    print("="*60)
    
    # Load data
    print("[1/6] -> Loading training Data...")
    train_data=pd.read_csv("data/splits/train.csv")
    val_data=pd.read_csv("data/splits/val.csv")
    
    # Create enviornment
    print("[2/6] -> Creating Enviornments...")
    train_env= TradingEnvironment(
        data=train_data,
        initial_balance=initial_balance,
        transaction_cost_pct=0.00001,
        shares_per_trade=5
    )
    val_env=TradingEnvironment(
        data=val_data,
        initial_balance=initial_balance,
        transaction_cost_pct=0.00001,
        shares_per_trade=5
    )
    
    print(f"State Dimensions:{train_env.observation_space.shape[0]}")
    print(f"Action Dimension:{train_env.action_space.n}")
    
    # Create agent
    print("[3/6] Initializing DQN Agent...")
# In src/train.py, when creating agent:

    agent = DQNAgent(
    state_dim=13,
    action_dim=3,
    learning_rate=0.0001,
    gamma=0.99,                   # Higher discount
    epsilon_start=1.0,
    epsilon_end=0.05,             # Some exploration kept
    epsilon_decay=0.995,          # Standard decay
    buffer_capacity=10000,
    batch_size=64,                # Larger batch
    target_update_freq=1000
)
    print(f"Network parameters:{sum(p.numel() for p in agent.main_network.parameters())}")
    print(f"Initial Epsilon:{agent.epsilon:.2f}")
    
    # Training metrics storage
    print("\n[4/6] Setting up metrics tracking...")
    train_rewards = []
    train_losses = []
    val_rewards = []
    epsilon_history = []

    # Training loop
    print("\n[5/6] Starting training...")
    print(f"  Episodes: {num_episodes}")
    print(f"  Target update frequency: {agent.target_update_freq} steps")
    print("-"*60)
    
    best_val_reward = -float('inf')

    for episode in range(num_episodes):
        state,_=train_env.reset()
        episode_reward=0
        episode_losses=[]
        done=False
        # Inside train.py, in the episode loop:
        if episode == 0:
            print(f"Buffer size after episode 0: {len(agent.replay_buffer)}")
        if episode == 10:
            print(f"Buffer size after episode 10: {len(agent.replay_buffer)}")
        while not done:
            # Select action
            action=agent.select_action(state=state)
            # Take action
            next_state, reward, done, truncated, info=train_env.step(action=action)
            
            if train_env.current_step < 3:
                print(f"Step {train_env.current_step}: action={action}, reward={reward:.4f}, "
                    f"portfolio={info['portfolio_value']:.2f}, done={done}")
            
            # Remember action
            agent.remember(state,action,reward,next_state,done)
            

            # Train agent
            loss=agent.train()
            if loss is not None:
                episode_losses.append(loss)
                
            state=next_state
            episode_reward+=reward
        agent.decay_eplsilon()
        
        
        # Store metrics
        train_rewards.append(episode_reward)
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        train_losses.append(avg_loss)
        epsilon_history.append(agent.epsilon)                

        # Validation Episode(every 10 episode)
        if (episode+1)%10 == 0:
            val_reward=evaluate_agent(agent,val_env)
            val_rewards.append(val_reward)
            
            if val_reward>best_val_reward:
                best_val_reward=val_reward
                agent.save('models/checkpoints/best_agent.pth')
                print(f"New best agent saved! Val reward {val_reward:.2f}")
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Train Reward: {episode_reward:.4f} | "
                  f"Avg Loss: {avg_loss:.6f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Val Reward: {val_reward:.4f}")        
        
    # Save final model
    print("\n[6/6] Training complete!")
    agent.save('models/checkpoints/final_agent.pth')
    print(f"  Final model saved")
    print(f"  Best validation reward: {best_val_reward:.4f}")            
    
    # Plot training curves
    print("\nGenerating training plots...")
    plot_training_curves(train_rewards, train_losses, val_rewards, epsilon_history)
    
    return agent, {
        'train_rewards': train_rewards,
        'train_losses': train_losses,
        'val_rewards': val_rewards,
        'epsilon_history': epsilon_history
    }
    
def evaluate_agent(agent,env,num_of_episodes=1):
    """
    Evaluate agent on enviornment(no exploration)
    
    Args:
        agent: Description
        env: Description
        num_of_episodes: Description
    Returns:
        avg_reward: Average reward over all episodes
    
    """
    step_count=0
    total_rewards=[]
    for _ in range(num_of_episodes):
        state,_= env.reset()
        episode_reward=0
        done=False
        
        while not done:
            action=agent.select_action(state,evaluation=True)
            next_state, reward, done, truncated, info=env.step(action)
            step_count += 1  # ADD THIS
            
            # ADD THIS DEBUG
            if step_count < 5:
                print(f"  Val step {step_count}: reward={reward:.6f}, done={done}")
        
            episode_reward+=reward
            state=next_state
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

def plot_training_curves(train_rewards, train_losses, val_rewards, epsilon_history):
    """Create training visualization plots"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Training rewards
    axes[0, 0].plot(train_rewards, alpha=0.6, color='blue')
    window = 20
    if len(train_rewards) >= window:
        ma = pd.Series(train_rewards).rolling(window).mean()
        axes[0, 0].plot(ma, color='red', linewidth=2, label=f'{window}-episode MA')
    axes[0, 0].set_title('Training Rewards per Episode')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Training loss
    axes[0, 1].plot(train_losses, alpha=0.6, color='green')
    if len(train_losses) >= window:
        ma_loss = pd.Series(train_losses).rolling(window).mean()
        axes[0, 1].plot(ma_loss, color='red', linewidth=2, label=f'{window}-episode MA')
    axes[0, 1].set_title('Training Loss per Episode')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Validation rewards
    val_episodes = list(range(10, len(train_rewards) + 1, 10))
    axes[1, 0].plot(val_episodes, val_rewards, marker='o', color='purple')
    axes[1, 0].set_title('Validation Rewards')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Validation Reward')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Epsilon decay
    axes[1, 1].plot(epsilon_history, color='orange')
    axes[1, 1].set_title('Exploration Rate (Epsilon) Decay')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Epsilon')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        'results/training_curves/training_progress.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)   # <<< IMPORTANT

    print("Saved to results/training_curves/training_progress.png")    

# Main entry point
if __name__ == "__main__":
    # Train agent
    agent, metrics = train_agent(
        num_episodes=500,
        initial_balance=100000
    )
    
    print("\nTraining summary:")
    print(f"  Final training reward: {metrics['train_rewards'][-1]:.4f}")
    print(f"  Best validation reward: {max(metrics['val_rewards']):.4f}")
    print(f"  Final epsilon: {metrics['epsilon_history'][-1]:.4f}")        
            
