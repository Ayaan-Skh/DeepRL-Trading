import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
# from src.logger import logging

class TradingEnvironment(gym.Env):
    """
    Custom Trading Environment that follows OpenAi Gym interface.
    
        State:[price, indicators, portfolio_info]
        Actions:[Hold:0,Buy:1,Sell:2]
        Reward: Adjusted portfolio returns
    
    """
    def __init__(self,
                 data,
                 initial_balance=100000,
                 transaction_cost_pct=0.0001,
                 shares_per_trade=10):
        """
        Initialize the trading environment
        
        Args:
            data: DataFrame with OHLCV + technical indicators
            initial_balance: Starting cash (e.g., ₹1,00,000)
            transaction_cost_pct: Trading fee (0.1% = 0.001)
            shares_per_trade: How many shares to buy/sell per action
        """
        super(TradingEnvironment, self).__init__()
        # Store data
        self.data=data.reset_index(drop=True)
        self.max_steps=len(data)-1
        
        # Trading parameters
        self.initial_balance=initial_balance
        self.transaction_cost_pct=transaction_cost_pct
        self.shares_per_trade=shares_per_trade
        
        self.action_space=spaces.Discrete(3) # 0:Hold,1:Buy,2:Sell

        print(f"Environment initialized: balance={self.initial_balance}, "
      f"shares_per_trade={self.shares_per_trade}")
        # Define observation Space (state space)
        # 13 continuous values: Open, High, Low, Close, Volume, SMA_20, SMA_50, RSI_14, MACD, MACD_Signal, BB_Upper, BB_Middle, BB_Lower
        self.observation_space=spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),
            dtype=np.float32
        )
        # State is a vector of 13 numbers
            # Each can be any real number (-∞ to +∞)
            # shape=(13,) means 1D array with 13 elements
            # dtype=np.float32 means 32-bit floating point
            
        # Initialize state variables (will be set in reset())
        self.current_step = 0
        self.balance = 0
        self.shares_held = 0
        self.portfolio_value = 0
        self.portfolio_history = []    
        
    def reset(self,seed =None):
        """
        Reset the environment to initial state
        
            What super().reset(seed=seed) does:
            Sets random seed for reproducibility (if provided).
        """
        super().reset(seed=seed)    
        # logging.info(f"Enviornment reset with seed:{seed}")
        self.current_step=0
        self.balance=self.initial_balance
        self.shares_held=0
        self.portfolio_value=self.initial_balance
        self.portfolio_history=[self.initial_balance]
        self._last_action=0
        
        print(f"Reset: balance={self.balance}, shares={self.shares_held}")

        return self._get_observations(), {}
        
    def _normalize_state(self, state):
        """`
        Normalize state features to prevent exploding Q-values
        
        Each feature scaled to roughly [-1, 1] or [0, 1] range
        """
        normalized = state.copy()
        
        # Price features (divide by typical price ~2000)
        normalized[0] = state[0] / 2000.0       # current_price
        normalized[1] = state[1] / 10.0         # price_change_pct (already %)
        normalized[2] = state[2] / 2000.0       # sma_20
        normalized[3] = state[3] / 2000.0       # sma_50
        
        # Technical indicators (already in reasonable ranges, just scale)
        normalized[4] = state[4] / 100.0        # rsi (0-100 → 0-1)
        normalized[5] = state[5] / 50.0         # macd (typically -50 to 50)
        normalized[6] = state[6] / 50.0         # macd_signal
        # normalized[7] already 0-1              # bb_percent
        
        # Portfolio features (divide by initial balance ~100000)
        normalized[8] = state[8] / 100000.0     # balance
        normalized[9] = state[9] / 100.0        # shares_held (typically 0-100)
        normalized[10] = state[10] / 100000.0   # portfolio_value
        # normalized[11] already 0-1             # shares_value_pct
        # normalized[12] already 0-1             # can_buy
        
        return normalized  
        
    def _get_observations(self):
        """
        Construct state vector from current data
        
        Returns:
            state: numpy array of 13 features
        """
        # logging.info(f"Getting observations at step:{self.current_step}")
        # Get current row of data    
        row= self.data.iloc[self.current_step]
        
        # Extract price features 
        current_price=row['Close']
        price_change=row['Price_Change_Pct']    
        # logging.info(f"Extracted price features: current_price={current_price}, price_change={price_change}")
        # Extract technical indicators
        sma_20=row['SMA_20']
        sma_50=row['SMA_50']
        rsi=row['RSI_14']
        macd=row['MACD']
        macd_signal=row['MACD_Signal']
        # logging.info(f"Extracted technical indicators: SMA_20={sma_20}, SMA_50={sma_50}, RSI_14={rsi}, MACD={macd}, MACD_Signal={macd_signal}")
        
        # Extract Bollinger band positions 
        bb_upper=row['BB_Upper']
        bb_lower=row['BB_Lower']
        bb_middle=row['BB_Middle']
        # logging.info(f"Extracted Bollinger Bands: BB_Upper={bb_upper}, BB_Lower={bb_lower}, BB_Middle={bb_middle}")
        # Normalize to 0-1 range
        if bb_upper != bb_lower:
            bb_percent=(current_price-bb_lower)/(bb_upper-bb_lower)   
        else:
            bb_percent=0.5 
            
            
        # Portfolio features
        shares_value = self.shares_held * current_price
        portfolio_value = self.balance + shares_value
        shares_value_pct = shares_value / portfolio_value if portfolio_value > 0 else 0  
        
        # Can we afford to buy?
        can_buy = 1.0 if self.balance >= current_price * self.shares_per_trade else 0.0
        # In _get_observation(), add:
        if self.current_step < 5:
            print(f"Step {self.current_step}: balance={self.balance:.0f}, "
                f"shares={self.shares_held}, can_buy={can_buy}")
        
        # Construct state vector
        state = np.array([
            current_price,
            price_change,
            sma_20,
            sma_50,
            rsi,
            macd,
            macd_signal,
            bb_percent,
            self.balance,
            self.shares_held,
            portfolio_value,
            shares_value_pct,
            can_buy
        ], dtype=np.float32)
        
        state=self._normalize_state(state)
        return state
    
    def step(self, action):
        """
        Execute one action and return results
        
        action: 0=Hold,1=Buy, 2=Sell
        
        Returns:
            next_state: Next observation
            reward: Reward for this observation
            done: Is the episode done?
            truncated: DId we run out of time
            info: extra info ()
        """
        self._last_action=action
        # Get current price
        current_price=self.data.iloc[self.current_step]['Close']
        
        # Store previous portfolio value (for reward calculation)
        prev_portfolio_value = self.portfolio_value
        
        if action==1:
            # Buy Shares
            self._execute_buy(current_price)
        elif action==2:
            # Sell Shares
            self._execute_sell(current_price)    
        #action==0 hold: Do nothing
        
        # Move to next step    
        self.current_step+=1
        done=self.current_step>=self.max_steps
        
        # if not done calculate current portfolio value
        if not done:
            current_price=self.data.iloc[self.current_step]['Close']
            
        # Calculate value of shares held
        shares_value=self.shares_held*current_price
        
        # Calculate the portfolio value
        self.portfolio_value=self.balance+shares_value
        if self.portfolio_value < self.initial_balance * 0.1:  # Lost 90%
            done = True
            reward = -1.0 
        
        # Append the calculated value to history
        self.portfolio_history.append(self.portfolio_value)    
        
        # Calculate reward
        reward=self._calculate_reward(prev_portfolio_value)
        
        # Get next state
        next_state = self._get_observations() if not done else np.zeros(13)
        
        # Gym requires 5 return values
        truncated = False  # We don't truncate episodes early
        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'shares_held': self.shares_held
        }
        
            # next_state: What you see now
            # reward: Feedback for last action
            # done: Is episode over?
            # truncated: Did we stop early? (No for us)
            # info: Extra debugging info (optional)
        return next_state, reward, done, truncated, info
    
    def _execute_buy(self,current_price):
        """
        Execute Buy action
        
        current_price: Current portfolio value
        
        """    
        shares_to_buy=self.shares_per_trade
        total_cost=shares_to_buy*current_price
        
        transction_fee=total_cost*self.transaction_cost_pct
        total_cost_with_fee=transction_fee+total_cost
        
        # Check if we can afford it
        if self.balance >= total_cost_with_fee:
            self.balance -= total_cost_with_fee
            self.shares_held += shares_to_buy
        else:
            return
        
    def _execute_sell(self,current_price):
        # Execute a sell action by liquidating shares at the current market price.
        # This method sells a portion of the investor's held shares up to the maximum
        # shares_per_trade limit. The actual number of shares sold is constrained by
        # the lesser of: the configured shares_per_trade amount or the total shares
        # currently held in the portfolio.
        # Args:
        #     current_price (float): The current market price per share at which to execute the sell.
        # Returns:
        #     None
        # Side Effects:
        #     - Reduces self.shares_held by the number of shares sold
        #     - Updates portfolio cash balance with proceeds from the sale
        #     - Records the transaction in trading history
        # Note:
        #     The variable shares_to_sell represents the quantity to be sold in this transaction.
        #     It is bounded by both trading limits and actual holdings to prevent overselling.
        """
        Execute the sell action
        
        :current_price: current price 
        """    
        if self.shares_held==0:
            return
        # Calculate shares to sell
        shares_to_sell=min(self.shares_per_trade,self.shares_held)

        if shares_to_sell>0:
            
            # Calculate proceeds from the sale
            total_proceeds=shares_to_sell*current_price

            # Substract transction fees from process
            transction_fee=self.transaction_cost_pct*total_proceeds           
            total_proceeds_after_fee=total_proceeds-transction_fee
            # logging.info(f"Selling {shares_to_sell} shares at price:{ current_price}, total proceeds after fee:{total_proceeds_after_fee}")
            
            ## Update portfolio
            self.balance-= total_proceeds_after_fee
            self.shares_held-=shares_to_sell
            # logging.info(f"Updated balance:{self.balance}, shares held:{self.shares_held}")
            
            # Updates:
            # Add cash to balance
            # Remove shares from holdings
            # If no shares to sell, action is ignored
        
    def _calculate_reward(self, prev_portfolio_value):
        """
        Reward function that HEAVILY encourages trading
        """
        # Portfolio return
        if prev_portfolio_value > 0:
            portfolio_return = (
                (self.portfolio_value - prev_portfolio_value) 
                / prev_portfolio_value
            )
        else:
            portfolio_return = 0
        
        # AGGRESSIVE: Only reward if traded
        if hasattr(self, '_last_action'):
            if self._last_action == 0:  # HOLD
                # HOLD gets NO reward, only penalty
                reward = -0.0001
            else:  # BUY or SELL
                # Trading gets portfolio return + small bonus
                reward = portfolio_return + 0.0001
        else:
            reward = portfolio_return
        
        return reward
        # Simple reward: just the return
        # Don't scale by 100 - that was too much!
        # reward = portfolio_return
        
        # return reward    # Interpretation:
            #     Made 2% return
            #     But with 1.5% volatility (risky)
            #     Net reward = 1.25% (decent!)
        
             
        
        
        
        
        
        
        
              