import sys
sys.path.append(".")

import pandas as pd
import numpy as np

from src.agent import DQNAgent
from src.enviornments import TradingEnvironment

import matplotlib
matplotlib.use("Agg")   
import matplotlib.pyplot as plt

def run_episode(env,agent=None,policy="random"):
    """
    
    Run one complete episode and track everything
    
    Args:    
        env: Trading enviornment
        agent: DQNAgent (If using trained policy)
        policy: "random","trained","buy_and_hold","always_hold"
    
    Returns:
        Dictionary with episodes metrices and history
    """
    state,_ = env.reset()
    
    # Storage
    portfolio_values=[env.portfolio_value]
    actions_taken=[]
    rewards_received=[]
    balances=[]
    shares_held=[env.shares_held]
    prices=[env.data.iloc[0]['Close']]
    
    done=False
    step=0

    while not done:
        if policy == 'random':
            action = np.random.choice([0,1,2])
        elif policy == 'trained':
            action = agent.select_action(state,evaluation=True)    
        elif policy == 'buy_and_hold':
            # Buy first then hold
            action = 1 if step==0 else 0    
        elif policy == 'always_hold':
            action=0
        else:
            action=0
        
        # Execute action        
        next_state,reward,done,truncated,info=env.step(action)
        
        # Append the record
        actions_taken.append(action)
        rewards_received.append(reward)
        portfolio_values.append(info['portfolio_value'])
        balances.append(info['balance'])
        shares_held.append(info['shares_held'])                            
        
        if not done:
            prices.append(env.data.iloc[env.current_step]['Close'])
            
        state=next_state
        step+=1
    
    # Calculate metrics    
    initial_value=portfolio_values[0]
    final_value=portfolio_values[-1]
    total_return=(final_value-initial_value)/initial_value    
    
    # Calculate the sharpe ratio
    returns=np.diff(portfolio_values)/portfolio_values[:-1]
    sharpe_ratio=np.mean(returns)/np.std(returns) * np.std(252) if np.std(returns)> 0 else 0
    
    # Calculate Maximum Drawdown
    peak=np.maximum.accumulate(portfolio_values)
    drawdown=(np.array(portfolio_values)-peak)/peak
    max_drawdown=np.min(drawdown)
    
    # WIn rates (Profitable trades)
    buy_indices = [i for i, a in enumerate(actions_taken) if a == 1]
    sell_indices = [i for i, a in enumerate(actions_taken) if a == 2]
    
    profitable_trades = 0
    total_trades = 0                    
        
    # Simple approximation: count trades where portfolio increased
    if len(sell_indices) > 0:
        for sell_idx in sell_indices:
            if sell_idx > 0:
                if portfolio_values[sell_idx] > portfolio_values[sell_idx-1]:
                    profitable_trades += 1
                total_trades += 1
    
    win_rate=profitable_trades/total_trades if total_trades > 0 else 0
    
    return{
        "policy":policy,
        "initial_value":initial_value,
        "final_value":final_value,
        "total_return":total_return,
        "total_return_pct":total_return*100,
        "sharpe_ratio":sharpe_ratio,
        "max_drawdown":max_drawdown,
        "max_drawdown_pct":max_drawdown*100,
        "win_rate":win_rate,
        "win_rate_pct":win_rate*100,
        "total_reward":sum(rewards_received),
        "total_steps":len(actions_taken),
        "num_buys":actions_taken.count(1),
        "num_sells":actions_taken.count(2),
        "num_holds":actions_taken.count(0),
        "portfolio_values":portfolio_values,
        "actions":actions_taken,
        "balances":balances,
        "Shares":shares_held,
        "prices":prices
        
    }                
        
def backtest_all_strategies():
    """
        Backtest All strategies on test set and compare
    """        
    print("="* 60)
    print("BACKTESTING ON TEST SET(2022-2023)....")
    print("="* 60)
    
    # Load dataset
    test_data=pd.read_csv("data/splits/test.csv")
    print(f"\n Test peroid: {test_data['Date'].min()} to {test_data['Date'].max()}")
    print(f"Test days:{len(test_data)}")
    
    # Create enviornment
    env=TradingEnvironment(
        data=test_data,
        initial_balance=1000000,
        shares_per_trade=10,
        transaction_cost_pct=0.0001
    )
    
    # Load trained agent
    agent=DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )
    agent.load("models/checkpoints/best_agent.pth")
    
    #Running all multiple times for stability
    strategies = {
        'Random': 'random',
        'Buy-and-Hold': 'buy_and_hold',
        'Always Hold': 'always_hold',
        'Trained Agent': 'trained'
    }
    
    results={}
    
    for name, policy in strategies.items():
        print(f"Testing {name} policy")
        if policy== "random":
            episodes=[]
            for _ in range(10):
                result=run_episode(env,agent,policy)
                episodes.append(result)
            
            # Average the results
            avg_results={
                'policy': name,
                'total_return_pct': np.mean([e['total_return_pct'] for e in episodes]),
                'sharpe_ratio': np.mean([e['sharpe_ratio'] for e in episodes]),
                'max_drawdown_pct': np.mean([e['max_drawdown_pct'] for e in episodes]),
                'win_rate_pct': np.mean([e['win_rate_pct'] for e in episodes]),
                'final_value': np.mean([e['final_value'] for e in episodes])
            }                
            results[name]=avg_results
            
            # Keep episode for plotting
            results[name]['episode_data']=episodes[0]
            
        else:
            # Run once for deterministic strategies
            result=run_episode(env,agent,policy)
            results[name]=result
            results[name]['episode_data']=result
    
    return results,test_data        
            
def print_results_table(results):
    """Print comparison table"""
    print("\n" + "="*80)
    print("BACKTEST RESULTS COMPARISON")
    print("="*80)
    print(f"{'Strategy':<20} {'Return':<12} {'Sharpe':<10} {'Max DD':<12} {'Win Rate':<12} {'Final Value':<15}")
    print("-"*80)
    
    for name in ['Random', 'Always Hold', 'Buy-and-Hold', 'Trained Agent']:
        r = results[name]
        print(f"{name:<20} "
              f"{r['total_return_pct']:>10.2f}% "
              f"{r['sharpe_ratio']:>10.3f} "
              f"{r['max_drawdown_pct']:>10.2f}% "
              f"{r['win_rate_pct']:>10.1f}% "
              f"₹{r['final_value']:>13,.0f}")
    
    print("="*80)
    
    # Highlight best performer
    best_return = max(results.values(), key=lambda x: x['total_return_pct'])
    print(f"\n🏆 Best Return: {best_return['policy']} ({best_return['total_return_pct']:.2f}%)")
    
    best_sharpe = max(results.values(), key=lambda x: x['sharpe_ratio'])
    print(f"🏆 Best Risk-Adjusted: {best_sharpe['policy']} (Sharpe: {best_sharpe['sharpe_ratio']:.3f})")
            
def plot_comparison(results, test_data):
    """Create comparison visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Portfolio Value Over Time
    ax = axes[0, 0]
    for name in ['Random', 'Buy-and-Hold', 'Trained Agent']:
        data = results[name]['episode_data']
        steps = range(len(data['portfolio_values']))
        ax.plot(steps, data['portfolio_values'], label=name, linewidth=2, alpha=0.8)
    
    ax.axhline(y=1000000, color='red', linestyle='--', alpha=0.5, label='Initial')
    ax.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Portfolio Value (₹)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Stock Price (for context)
    ax = axes[0, 1]
    prices = results['Buy-and-Hold']['episode_data']['prices']
    ax.plot(prices, color='black', linewidth=2, alpha=0.7)
    ax.set_title('Stock Price (RELIANCE)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Price (₹)')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Returns Comparison (Bar chart)
    ax = axes[1, 0]
    names = ['Random', 'Always Hold', 'Buy-and-Hold', 'Trained Agent']
    returns = [results[name]['total_return_pct'] for name in names]
    colors = ['red' if r < 0 else 'green' for r in returns]
    
    bars = ax.bar(names, returns, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_title('Total Return Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Return (%)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, ret in zip(bars, returns):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{ret:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
    
    # Plot 4: Metrics Comparison (Radar chart alternative: grouped bars)
    ax = axes[1, 1]
    
    metrics_data = {
        'Sharpe': [results[name]['sharpe_ratio'] for name in names],
        'Win Rate': [results[name]['win_rate_pct']/100 for name in names]  # Normalize to 0-1
    }
    
    x = np.arange(len(names))
    width = 0.35
    
    ax.bar(x - width/2, metrics_data['Sharpe'], width, label='Sharpe Ratio', alpha=0.8)
    ax.bar(x + width/2, metrics_data['Win Rate'], width, label='Win Rate (normalized)', alpha=0.8)
    
    ax.set_title('Risk Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/backtest_comparison.png', dpi=300, bbox_inches='tight')
    print("\n📊 Saved comparison plots to results/backtest_comparison.png")
    plt.show()            
            
            
def save_results_to_csv(results):
    """Save results to CSV for documentation"""
    data = []
    for name in ['Random', 'Always Hold', 'Buy-and-Hold', 'Trained Agent']:
        r = results[name]
        data.append({
            'Strategy': name,
            'Total Return (%)': f"{r['total_return_pct']:.2f}",
            'Sharpe Ratio': f"{r['sharpe_ratio']:.3f}",
            'Max Drawdown (%)': f"{r['max_drawdown_pct']:.2f}",
            'Win Rate (%)': f"{r['win_rate_pct']:.1f}",
            'Final Portfolio Value': f"₹{r['final_value']:,.0f}"
        })
    
    df = pd.DataFrame(data)
    df.to_csv('results/backtest_results.csv', index=False)
    print("📄 Saved results to results/backtest_results.csv")


if __name__ == "__main__":
    # Run complete backtest
    results, test_data = backtest_all_strategies()
    
    # Print results table
    print_results_table(results)
    
    # Create visualizations
    plot_comparison(results, test_data)
    
    # Save to CSV
    save_results_to_csv(results)
    
    print("\n✅ Backtesting complete!")
    print("Next: Build Streamlit dashboard to showcase results")            