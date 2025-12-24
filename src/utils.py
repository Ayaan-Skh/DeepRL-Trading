import numpy as np
import torch.nn as nn
import pandas as pd
import random 
import torch
import yaml
from pathlib import Path


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    '''What problem this solves

RL is stochastic by nature:

random actions

random sampling from replay buffer

random weight initialization

GPU nondeterminism

Without fixing seeds:

You can’t debug, compare runs, or trust results.'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    """
        Fixes randomness in:

        Python

        NumPy

        PyTorch (CPU)
    """
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    """
        Fixes randomness on GPU(s).
    """
    torch.backends.cudnn.deterministic =True
    torch.backends.cudnn.benchmark = False
    
    """
        This is important:

            Forces deterministic behavior in CUDA kernels

            Slightly slower, but reproducible
    """
    
def load_config(config_path="configs/config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_directories():
    """Create necessary directories if they don't exist"""
    dirs = [
        'data/raw',
        'data/processed',
        'data/splits',
        'models/checkpoints',
        'results/training_curves',
        'results/backtest_results',
        'results/logs'
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)    
        
def save_model(model,path,episode,optimizer=True):
    """Save model checkpoint"""
    checkpoint={
        "episode":episode,
        "model_state_dict":model.state_dict(),
        "optimizer":optimizer.state_dict() if optimizer else None
    }
    torch.save(checkpoint,path)
    print(f"Model saved to path:{path} at episode:{episode}")
    
def load_model(model,path,optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and checkpoint['optimizer_state_dict']:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['episode']    

def calculate_returns(prices):
    """Calculate percentage returns from price series"""
    return prices.pct_change().fillna(0)

def calculate_sharpe_ratio(returns,risk_free_rate=0.0,peroids=252):
    """
        Docstring for calculate_sharpe_ratio
        
        :param returns: pandas series of returns
        :param risk_free_rate: annual risk free rate(default 0.0)
        :param peroids: bumber of peroids per years
    """
    excess_returns=returns-risk_free_rate/peroids
    if excess_returns.std()==0:
        return 0.0
    return np.sqrt(peroids)*excess_returns.mean()/excess_returns.std()

def calculate_max_drawdown(portfolio_values):
    """
        Docstring for calculate_max_drawdown
        
        :param portfolio_values: Pandas series of portfolio values
    """
    cumulative_returns=portfolio_values/portfolio_values.iloc[0]
    running_max=cumulative_returns.expanding().max()
    drawdown=(cumulative_returns-running_max)/running_max
    return drawdown.min()

def epsilon_greedy_decay(episode,epsilon_start=1.0,epsilon_end=0.01,decay_rate=0.995):
    """Calculate epsilon for epsilon greedy exploration strategy"""
    return max(epsilon_end,epsilon_start*(decay_rate**episode))

class Logger:
    """Simple logger for training metrics"""
    
    def __init__(self, log_dir="results/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = []
    
    def log(self, episode, metrics_dict):
        """Log metrics for an episode"""
        metrics_dict['episode'] = episode
        self.metrics.append(metrics_dict)
    
    def save(self, filename="training_log.csv"):
        """Save metrics to CSV"""
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.log_dir / filename, index=False)
        print(f"Logs saved to {self.log_dir / filename}")
    
    def get_dataframe(self):
        """Return metrics as pandas DataFrame"""
        return pd.DataFrame(self.metrics)

#Quick test
if __name__=="__main__":
    set_seed(42)
    create_directories()
    
    #Test sharpe ratio calculations
    returns = pd.Series([0.01, -0.005, 0.02, 0.015, -0.01])
    sharpe = calculate_sharpe_ratio(returns)
    print(f"Sharpe Ratio: {sharpe:.3f}")
    
    # Test max drawdown
    portfolio = pd.Series([10000, 10500, 10200, 11000, 9500])
    max_dd = calculate_max_drawdown(portfolio)
    print(f"Max Drawdown: {max_dd:.2%}")
