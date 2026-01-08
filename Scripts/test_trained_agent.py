import sys
sys.path.append('.')

import pandas as pd
from src.enviornments import TradingEnvironment
from src.agent import DQNAgent

test_data = pd.read_csv('data/splits/test.csv')
env = TradingEnvironment(test_data, initial_balance=100000, shares_per_trade=5)

agent = DQNAgent(state_dim=13, action_dim=3)
agent.load('models/checkpoints/best_agent.pth')

# Track actions
state, _ = env.reset()
actions_taken = []
done = False

while not done:
    action = agent.select_action(state, evaluation=True)
    actions_taken.append(action)
    next_state, reward, done, truncated, info = env.step(action)
    state = next_state

# Count actions
action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
print("Action distribution:")
for action in [0, 1, 2]:
    count = actions_taken.count(action)
    pct = count / len(actions_taken) * 100
    print(f"  {action_names[action]}: {count} times ({pct:.1f}%)")