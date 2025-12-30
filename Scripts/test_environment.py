import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from src.enviornments import TradingEnvironment


def test_environment():
    #Load training data
    print("Loading data....")
    data=pd.read_csv('data/splits/train.csv')
    print(10*"="+"Data loading complete"+ 10*"=")
    
    
    # Create enviornment
    env=TradingEnvironment(
        data=data,
        initial_balance=100000,
        transction_cost_pct=0.001,
        shares_per_trade=10
        )
    print("\n" + "="*50)
    print("TESTING ENVIRONMENT")
    print("="*50)
    
     # Test 1: Reset
    print("\nTest 1: Reset environment")
    state, _ = env.reset()
    print(f"  Initial state shape: {state.shape}")
    print(f"  Initial state: {state}")
    print(f"  Initial balance: ₹{env.balance:,.2f}")
    print(f"  Initial shares: {env.shares_held}")
    
    
    print("\n Test 2: Taking random actions")
    total_reward=0
    
    for step in range(10):
        action=np.random.choice([0,1,2])
        action_name=['HOLD','BUY','SELL'][action]
        
        next_state,reward,done,truncated,info=env.step(action)
        total_reward += reward
        
        print(f"Step {step+1}:{action_name}")
        print(f"Portfolio value:{info['portfolio_value']:,.2f}")
        print(f"Balance: ₹{info['portfolio_value']:,.2f}")
        print(f"Shares: {info['shares_held']}")
        print(f"Reward: {reward:.6f}")
        
        if done:
            print("Episode ended!")
            break
    print(f"\n Total Reward over 10 steps: {total_reward:.6f}")
    
    # Test 3: Full Episode
    state,_= env.reset()
    episode_rewards=[]
    
    done-False
    while not done:
        action = np.random.choice([0, 1, 2])
        next_state, reward, done, truncated, info = env.step(action)
        episode_rewards.append(reward)
        state = next_state
        
    print(f"  Episode length: {len(episode_rewards)} steps")
    print(f"  Total return: {sum(episode_rewards):.4f}")
    print(f"  Final portfolio value: ₹{env.portfolio_value:,.2f}")
    print(f"  Profit/Loss: ₹{env.portfolio_value - env.initial_balance:,.2f}")   
    
    
      # Test 4: Action space and observation space
    print("\nTest 4: Space verification")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  State dimension: {env.observation_space.shape}") 
    
    print("\n" + "="*50)
    print("ALL TESTS PASSED! ✓")
    print("="*50)

if __name__ == "__main__":
    test_environment()
    
        
        