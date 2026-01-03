import sys
sys.path.append(".")

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")   # <<< FIX

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from src.enviornments import TradingEnvironment

def run_and_visualize_episode(policy='random',num_steps=100):
    """
    Run one episode and give ita visualization
    
        poilcy:"random" or "buy_and_hold" or "always_buy"
        num_steps: How many steps to run (Use less for clear plots)
    """
    data=pd.read_csv("data/splits/train.csv")
    
    # Creating enviornment
    env = TradingEnvironment(
        data=data,
        initial_balance=100000,
        transaction_cost_pct=0.001,
        shares_per_trade=10
    )
    state,_=env.reset()
    
    print(f"Initial shares:{env.shares_held}")
    
    print(f"Running episode with {policy} policy for {num_steps} steps...")
    
        # Storage for tracking
    states = []
    actions = []
    rewards = []
    portfolio_values = []
    prices = []
    balances = []
    shares = []
    
    #Reset enviornment
    state,_=env.reset()
    
    # Run episodes
    for step in range(min(num_steps,env.max_steps)):
        
        # Store current state
        states.append(state)
        prices.append(data.iloc[step]['Close'])
        
        # Select action based on policy
        if policy == 'random':
            action = np.random.choice([0, 1, 2])
        elif policy == 'buy_and_hold':
            # Buy on day 1, then hold forever
            action = 1 if step == 0 else 0
        elif policy == 'always_buy':
            action = 1  # Always try to buy
        elif policy == 'always_sell':
            action = 2  # Always try to sell
        else:
            action = 0  # Default to hold  

        next_state,reward,done,truncated,info=env.step(action)
        
        # Store results
        actions.append(action)
        rewards.append(reward)
        portfolio_values.append(info['portfolio_value'])
        balances.append(info.get('balance', 0))
        shares.append(info.get('shares_held', 0))

        # move to next state
        state=next_state
        
        if done:
            break
            
    print(f"Episode completed: {len(actions)} steps")
    print(f"Final portfolio value: ₹{portfolio_values[-1]:,.2f}")
    print(f"Total return: {(portfolio_values[-1] - 100000) / 100000 * 100:.2f}%")    
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(5, 1, figsize=(15, 12))
    steps = range(len(actions))
    
    # Plot 1: Portfolio Value Over Time
    axes[0].plot(steps, portfolio_values, linewidth=2, color='blue')
    axes[0].axhline(y=100000, color='red', linestyle='--', 
                    label='Initial Balance', alpha=0.7)
    axes[0].fill_between(steps, 100000, portfolio_values, 
                          where=np.array(portfolio_values) >= 100000,
                          alpha=0.3, color='green', label='Profit')
    axes[0].fill_between(steps, 100000, portfolio_values,
                          where=np.array(portfolio_values) < 100000,
                          alpha=0.3, color='red', label='Loss')
    axes[0].set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Portfolio Value (₹)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    
    # Plot 2: Stock Price
    axes[1].plot(steps, prices[:len(steps)], linewidth=2, color='black')
    axes[1].set_title('Stock Price', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Price (₹)')
    axes[1].grid(True, alpha=0.3)
    
    
    # Plot 3: Actions Taken
    action_colors = {0: 'gray', 1: 'green', 2: 'red'}
    action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
    
    for action_type in [0, 1, 2]:
        action_steps = [i for i, a in enumerate(actions) if a == action_type]
        action_values = [prices[i] for i in action_steps if i < len(prices)]
        axes[2].scatter(action_steps[:len(action_values)], action_values, 
                       c=action_colors[action_type], 
                       label=action_names[action_type],
                       alpha=0.6, s=50)
    
    axes[2].plot(steps, prices[:len(steps)], linewidth=1, 
                 color='black', alpha=0.3, label='Price')
    axes[2].set_title('Actions Taken', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Price (₹)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Cash Balance and Shares Held
    ax4_left = axes[3]
    ax4_right = ax4_left.twinx()
    
    ax4_left.plot(steps, balances, linewidth=2, color='green', label='Cash Balance')
    ax4_right.plot(steps, shares, linewidth=2, color='blue', label='Shares Held')
    
    ax4_left.set_ylabel('Cash Balance (₹)', color='green')
    ax4_right.set_ylabel('Shares Held', color='blue')
    ax4_left.set_title('Cash vs Shares Over Time', fontsize=14, fontweight='bold')
    ax4_left.grid(True, alpha=0.3)
    
    # Add legends
    lines1, labels1 = ax4_left.get_legend_handles_labels()
    lines2, labels2 = ax4_right.get_legend_handles_labels()
    ax4_left.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Plot 5: Cumulative Rewards
    cumulative_rewards = np.cumsum(rewards)
    axes[4].plot(steps, cumulative_rewards, linewidth=2, color='purple')
    axes[4].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[4].fill_between(steps, 0, cumulative_rewards,
                          where=np.array(cumulative_rewards) >= 0,
                          alpha=0.3, color='green')
    axes[4].fill_between(steps, 0, cumulative_rewards,
                          where=np.array(cumulative_rewards) < 0,
                          alpha=0.3, color='red')
    axes[4].set_title('Cumulative Rewards', fontsize=14, fontweight='bold')
    axes[4].set_ylabel('Cumulative Reward')
    axes[4].set_xlabel('Step')
    axes[4].grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = Path(f'results/plots/episode_{policy}.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {out_path}")

    # Only show plot if an interactive backend is available
    if not matplotlib.get_backend().lower().startswith('agg'):
        plt.show()
    else:
        plt.close(fig)
    
    # Print statistics
    print("\n" + "="*50)
    print("EPISODE STATISTICS")
    print("="*50)
    print(f"Policy: {policy}")
    print(f"Steps: {len(actions)}")
    print(f"Initial balance: ₹1,00,000.00")
    print(f"Final portfolio value: ₹{portfolio_values[-1]:,.2f}")
    print(f"Total return: {(portfolio_values[-1] - 100000) / 100000 * 100:.2f}%")
    print(f"Total reward: {sum(rewards):.4f}")
    print(f"\nAction distribution:")
    print(f"  HOLD: {actions.count(0)} ({actions.count(0)/len(actions)*100:.1f}%)")
    print(f"  BUY:  {actions.count(1)} ({actions.count(1)/len(actions)*100:.1f}%)")
    print(f"  SELL: {actions.count(2)} ({actions.count(2)/len(actions)*100:.1f}%)")

if __name__ == "__main__":
    # Test different policies
    print("Testing Random Policy:")
    run_and_visualize_episode(policy='random', num_steps=252)
    
    print("\n" + "="*50 + "\n")
    
    print("Testing Buy-and-Hold Policy:")
    run_and_visualize_episode(policy='buy_and_hold', num_steps=252)
    
    
    
    