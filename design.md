# DeepTrade Design Document

## 1. Problem Definition

**Goal:** Learn an optimal trading policy that maximizes risk-adjusted returns

**Why RL?**
- Sequential decision making (not just prediction)
- Actions affect future states (buy now → fewer shares to buy later)
- Learn from trial and error (no labeled "best action")

## 2. MDP Formulation

### State Space
```python
state = [
    # Price features (3)
    current_price,
    price_pct_change,
    price_volatility,
    
    # Technical indicators (6)
    sma_20,
    rsi_14,
    macd,
    macd_signal,
    bb_upper,
    bb_lower,
    
    # Portfolio state (4)
    cash_balance,
    shares_held,
    portfolio_value,
    position_ratio  # shares_value / total_portfolio
]
# Total: 13 features
```

### Action Space
```python
actions = {
    0: HOLD,   # No transaction
    1: BUY,    # Buy X shares (X = min(cash/price, max_shares))
    2: SELL    # Sell X shares (X = min(shares_held, max_shares))
}
```

### Reward Function
```python
# Option 1: Simple Return
r_t = (portfolio_value_t - portfolio_value_{t-1}) / portfolio_value_{t-1}

# Option 2: Risk-Adjusted (USING THIS)
r_t = portfolio_return_t - λ * portfolio_risk_t
where λ = 0.5 (risk aversion parameter)

# Option 3: Sharpe Ratio (Advanced)
r_t = (mean_return - risk_free_rate) / std_return
```

### Transition Dynamics
```python
# Deterministic portfolio updates
cash_{t+1} = cash_t - cost_of_purchase + proceeds_from_sale - transaction_cost
shares_{t+1} = shares_t + shares_bought - shares_sold

# Stochastic price updates (market driven)
price_{t+1} ~ Market(price_t, external_factors)
```

## 3. Agent Architecture

### DQN Network

##### Input (13) → Dense(64, ReLU) → Dense(64, ReLU) → Dense(32, ReLU) → Output(3)

### Key Components
- Experience Replay Buffer (capacity: 10k)
- Target Network (updated every 1000 steps)
- Double DQN (action selection decoupled from evaluation)
- Epsilon-greedy exploration (ε: 1.0 → 0.01)

## 4. Training Procedure

### Phase 1: Exploration (Episodes 1-100)
- High epsilon (0.7-1.0)
- Fill replay buffer
- Agent learns basic patterns

### Phase 2: Learning (Episodes 100-400)
- Decaying epsilon (0.7 → 0.1)
- Active learning from experience
- Q-values stabilize

### Phase 3: Exploitation (Episodes 400-500)
- Low epsilon (0.1 → 0.01)
- Fine-tuning policy
- Final performance evaluation

## 5. Evaluation Metrics

### Training Metrics
- Episode return (cumulative reward)
- Average Q-value
- Loss
- Epsilon
- Action distribution

### Test Metrics
- **Total Return:** (final_value - initial_value) / initial_value
- **Sharpe Ratio:** risk-adjusted returns
- **Max Drawdown:** worst peak-to-trough decline
- **Win Rate:** % of profitable trades

### Baselines
1. Random agent (random actions)
2. Buy-and-hold (buy at start, sell at end)
3. SMA Crossover (simple technical strategy)

## 6. Risk Considerations

### Overfitting Prevention
- Train/Val/Test split (70/15/15)
- Early stopping on validation performance
- Regularization (dropout if needed)

### Realistic Constraints
- Transaction costs (0.1% per trade)
- Limited shares per trade
- No short selling (for MVP)
- No margin trading

## 7. Success Criteria

**Minimum (MVP):**
- Agent learns (not random trading)
- Beats random baseline
- Sharpe ratio > 0

**Target:**
- Beats buy-and-hold on test set
- Sharpe ratio > 0.5
- Max drawdown < 20%

**Stretch:**
- Sharpe ratio > 1.0
- Beats SMA crossover strategy
- Generalizes to multiple stocks

## 8. Timeline

- Day 3: Data collection ✓ (target)
- Day 4-5: Environment ✓
- Day 6-7: Agent ✓
- Day 8-14: Training ✓
- Day 15-21: Evaluation ✓