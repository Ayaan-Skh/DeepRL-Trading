import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
from src.logger import logging

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
                 transction_cost_pct=0.001,
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
        self.transction_cost_pct=transction_cost_pct
        self.shares_per_trade=shares_per_trade
        
        self.action_space=spaces.Discrete(3) # 0:Hold,1:Buy,2:Sell

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
        logging.info(f"Enviornment reset with seed:{seed}")
        self.current_step=0
        self.balance=self.initial_balance
        self.shares_held=0
        self.portfolio_value=self.initial_balance
        self.portfolio_history=[self.initial_balance]
        return self._get_observations(), {}
        
        
        
    def _get_observations(self):
        """
        Construct state vector from current data
        
        Returns:
            state: numpy array of 13 features
        """
        logging.info(f"Getting observations at step:{self.current_step}")
        # Get current row of data    
        row= self.data.iloc[self.current_step]
        
        # Extract price features 
        current_price=row['Close']
        price_change=row['Price_Change_Pct']    
        logging.info(f"Extracted price features: current_price={current_price}, price_change={price_change}")
        # Extract technical indicators
        sma_20=row['SMA_20']
        sma_50=row['SMA_50']
        rsi=row['RSI_14']
        macd=row['MACD']
        macd_signal=row['MACD_Signal']
        logging.info(f"Extracted technical indicators: SMA_20={sma_20}, SMA_50={sma_50}, RSI_14={rsi}, MACD={macd}, MACD_Signal={macd_signal}")
        
        # Extract Bollinger band positions 
        bb_upper=row['BB_Upper']
        bb_lower=row['BB_Lower']
        bb_middle=row['BB_Middle']
        logging.info(f"Extracted Bollinger Bands: BB_Upper={bb_upper}, BB_Lower={bb_lower}, BB_Middle={bb_middle}")
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
        logging.info(f"Attempting to buy {shares_to_buy} shares at price:{current_price} with total cost:{total_cost}")
        
        transction_fee=total_cost*self.transction_cost_pct
        total_cost_with_fee=transction_fee+total_cost
        
        # Check if we can afford it
        if self.balance >= total_cost_with_fee:
            self.balance -= total_cost_with_fee
            self.shares_held += shares_to_buy
        
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
        # Calculate shares to sell
        shares_to_sell=min(self.shares_per_trade,self.shares_held)
        if shares_to_sell>0:
            
            # Calculate proceeds from the sale
            total_proceeds=shares_to_sell*current_price

            # Substract transction fees from process
            transction_fee=self.transction_cost_pct*total_proceeds           
            total_proceeds_after_fee=total_proceeds-transction_fee
            logging.info(f"Selling {shares_to_sell} shares at price:{ current_price}, total proceeds after fee:{total_proceeds_after_fee}")
            
            ## Update portfolio
            self.balance-= total_proceeds_after_fee
            self.shares_held-=shares_to_sell
            logging.info(f"Updated balance:{self.balance}, shares held:{self.shares_held}")
            
            # Updates:
            # Add cash to balance
            # Remove shares from holdings
            # If no shares to sell, action is ignored
        
    def _calculate_reward(self,prev_portfolio_value):
        """
         Calculate reward based on portfolio performance
        
        Args:
            prev_portfolio_value: Portfolio value before action
            
        Returns:
            reward: Scalar reward value
        """
        if prev_portfolio_value>0:
            portfolio_value=((self.portfolio_value-prev_portfolio_value)/prev_portfolio_value)
            logging.info(f"Calculated reward:{portfolio_value}")
        else:
            portfolio_value=0.0
        
        # Calculate volitility penelty over last 20 steps
        if len(self.portfolio_history)>20:
            recent_values=self.portfolio_history[-20:]
            returns=np.diff(recent_values)/recent_values[:-1]
            volitility=np.std(returns)
            logging.info(f"Calculated volitility penelty: {volitility}")
        else:
            volitility=0.0
            
        risk_penalty=0.5
        reward=portfolio_value-risk_penalty*volitility
        logging.info(f"Final reward with penalty{risk_penalty*volitility} is:{reward}")    
        return reward
        # Interpretation:
        #     Made 2% return
        #     But with 1.5% volatility (risky)
        #     Net reward = 1.25% (decent!)
        
             
        
        
        
        
        
        
        
              