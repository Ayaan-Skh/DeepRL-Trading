import sys
sys.path.append(".")

import numpy as np
import pandas as pd

from src.enviornments import TradingEnvironment

def test_edge_cases():
    """Test edge cases that migth break the enviornment"""
    data=pd.read_csv("data/splits/train.csv")
    
    print("="*50)
    print("EDGE CASE TESTING")
    print("="*50)
    
    # Test 1: Can't buy with insufficient funds
    print("\nTest 1: Trying to buy with no money")
    env = TradingEnvironment(
            data=data, 
            initial_balance=100, 
            shares_per_trade=10
        )
    state, _ = env.reset()
    
    print(f"Initial balance: ₹{env.balance:.2f}")
    print(f"Stock price: ₹{data.iloc[0]['Close']:.2f}")
    print(f"Can afford 10 shares? {env.balance >= data.iloc[0]['Close'] * 10}")
    
    next_state, reward, done, truncated, info = env.step(1)  # BUY
    
    print(f"After BUY action:")
    print(f"  Balance: ₹{info['balance']:.2f}")
    print(f"  Shares: {info['shares_held']}")
    print(f"  Reward: {reward:.6f}")
    
    assert info['balance'] == 100, "Balance should not change if can't afford!"
    assert info['shares_held'] == 0, "Should not get shares if can't afford!"
    print("✓ Passed: Can't buy with insufficient funds")

    # Test case 2: Cant sell with no shares
    
    print("\n Test 2: Trying to sell with no shares")
    env = TradingEnvironment(
        data=data,
        initial_balance=100000,
        transaction_cost_pct=0.001,
        shares_per_trade=10
    )
    state,_=env.reset()
    print(f"Initial shares: {env.shares_held}")

    next_state, reward,done,truncated,info=env.step(2)
    
    print(f"After sell action")
    print(f"Balance ₹ {info['balance']:.2f}")
    print(f"Shares:{env.shares_held}")
    print(f"Reward:{reward:.6f}")
    
    
    assert info["balance"]==100000, "Balance should not be change if the shares are 0"
    assert info["shares_held"]==0, " Shares shoule still be 0!"
    print(f"✓ Passed: Cant sell whithout shares")
    
    #This tries when agent tries to sell with nos shares
    
    # Test 3: To see if the sate values are reasonable
    print(f"\n Test 3: State values are within expected ranges")
    env=TradingEnvironment(
        data=data,
        initial_balance=100000,
        shares_per_trade=10
    )
    print(f"State vector analysis")
    print(f"Price (state[0]): ₹{state[0]:.2f}")
    print(f"  Price change (state[1]): {state[1]:.2f}%")
    print(f"  RSI (state[4]): {state[4]:.2f} (should be 0-100)")
    print(f"  Balance (state[8]): ₹{state[8]:.2f}")
    print(f"  Shares (state[9]): {state[9]:.0f}")
    print(f"  Can buy flag (state[12]): {state[12]:.0f} (should be 0 or 1)")
    assert 0 <= state[4] <= 100, f"RSI should be 0-100, got {state[4]}"
    
    # Validate can_buy flag
    assert state[12] in [0.0, 1.0], f"can_buy should be 0 or 1, got {state[12]}"
    
    print("✓ Passed: State values are reasonable")
    
    # Test 4: Transaction costs are applied
    print("\nTest 4: Transaction costs are deducted")
    env = TradingEnvironment(
        data=data, 
        initial_balance=100000, 
        transaction_cost_pct=0.001,  # 0.1%
        shares_per_trade=10
    )
    state, _ = env.reset()
    
    initial_balance = env.balance
    stock_price = data.iloc[0]['Close']
    
    # Calculate expected cost
    shares_cost = 10 * stock_price
    transaction_fee = shares_cost * 0.001
    expected_cost = shares_cost + transaction_fee
    
    print(f"Stock price: ₹{stock_price:.2f}")
    print(f"Shares cost (10 × price): ₹{shares_cost:.2f}")
    print(f"Transaction fee (0.1%): ₹{transaction_fee:.2f}")
    print(f"Expected total cost: ₹{expected_cost:.2f}")
    
    # Buy shares
    next_state, reward, done, truncated, info = env.step(1)  # BUY
    
    actual_cost = initial_balance - info['balance']
    print(f"Actual cost deducted: ₹{actual_cost:.2f}")
    
    # Allow small floating point error
    assert abs(actual_cost - expected_cost) < 0.01, \
        f"Cost mismatch! Expected {expected_cost}, got {actual_cost}"
    
    print("✓ Passed: Transaction costs are applied correctly")    
    
    
    # Test 5: Episode ends at max steps
    print("\nTest 5: Episode terminates correctly")
    small_data = data.head(10)  # Only 10 days of data
    # print(pd.DataFrame(data=small_data))
    env = TradingEnvironment(data=small_data, initial_balance=100000)
    
    state, _ = env.reset()
    
    steps_taken = 0
    done = False
    
    while not done:
        next_state, reward, done, truncated, info = env.step(0)  # HOLD
        steps_taken += 1
        
        if steps_taken > 20:  # Safety check
            raise Exception("Episode should have ended!")
    steps_taken+=1
    print(f"Episode ended after {steps_taken} steps")
    assert steps_taken == 10, f"Expected 10 steps, got {steps_taken}"
    print("✓ Passed: Episode terminates at max steps")
    
    # Test 6: Reward calculation
    print("\nTest 6: Reward function produces reasonable values")
    env = TradingEnvironment(data=data, initial_balance=100000, shares_per_trade=10)
    state, _ = env.reset()
    
    rewards_list = []
    for _ in range(50):
        action = np.random.choice([0, 1, 2])
        next_state, reward, done, truncated, info = env.step(action)
        rewards_list.append(reward)
        
        if done:
            break
    
    print(f"Collected {len(rewards_list)} rewards")
    print(f"Reward range: [{min(rewards_list):.6f}, {max(rewards_list):.6f}]")
    print(f"Mean reward: {np.mean(rewards_list):.6f}")
    print(f"Std reward: {np.std(rewards_list):.6f}")
    
    # Rewards should be reasonable (not infinity, not too large)
    assert all(abs(r) < 1.0 for r in rewards_list), \
        "Rewards should typically be < 1.0 (since returns are percentages)"
    
    print("✓ Passed: Rewards are in reasonable range")
    
    print("\n" + "="*50)
    print("ALL EDGE CASE TESTS PASSED! ✓")
    print("="*50)


if __name__=="__main__":
    test_edge_cases()
