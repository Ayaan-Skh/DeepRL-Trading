import sys
sys.path.append('.')

import pandas as pd
from src.enviornments import TradingEnvironment
from src.agent import DQNAgent

test_data = pd.read_csv('data/splits/test.csv')
env = TradingEnvironment(test_data, initial_balance=1000000, shares_per_trade=10, transaction_cost_pct=0.0001)

agent = DQNAgent(state_dim=13, action_dim=3)
agent.load('models/checkpoints/best_agent.pth')

state, _ = env.reset()
actions = []
portfolios = []
balances = []
shares = []

done = False
step = 0

while not done:
    action = agent.select_action(state, evaluation=True)
    actions.append(action)
    
    # Print first 20 steps
    if step < 20:
        print(f"Step {step}: action={action} ({'HOLD' if action==0 else 'BUY' if action==1 else 'SELL'}), "
              f"balance=₹{env.balance:.0f}, shares={env.shares_held}, portfolio=₹{env.portfolio_value:.0f}")
    
    next_state, reward, done, truncated, info = env.step(action)
    
    portfolios.append(info['portfolio_value'])
    balances.append(info['balance'])
    shares.append(info['shares_held'])
    
    state = next_state
    step += 1

print(f"\n{'='*60}")
print(f"HOLD: {actions.count(0)} ({actions.count(0)/len(actions)*100:.1f}%)")
print(f"BUY:  {actions.count(1)} ({actions.count(1)/len(actions)*100:.1f}%)")
print(f"SELL: {actions.count(2)} ({actions.count(2)/len(actions)*100:.1f}%)")
print(f"\nFinal: balance=₹{balances[-1]:.0f}, shares={shares[-1]}, portfolio=₹{portfolios[-1]:.0f}")
print(f"Initial portfolio: ₹1,000,000")
print(f"Final portfolio: ₹{portfolios[-1]:,.0f}")
print(f"Change: ₹{portfolios[-1] - 1000000:,.0f} ({(portfolios[-1]/1000000 - 1)*100:.2f}%)")