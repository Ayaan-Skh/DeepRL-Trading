
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
sys.path.append('.')

from src.enviornments import TradingEnvironment
from src.agent import DQNAgent

# Page config
st.set_page_config(
    page_title="DeepTrade: RL Trading Agent",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("DeepTrade: Reinforcement Learning Trading Agent")
st.markdown("*A Deep Q-Network (DQN) agent trained to optimize portfolio trading on Indian stock market*")

# Sidebar
st.sidebar.header("Configuration")

# Stock selector
stock = st.sidebar.selectbox(
    "Select Stock",
    ["RELIANCE.NS (Trained)", "TCS.NS (Coming Soon)", "INFY.NS (Coming Soon)"],
    help="Currently only RELIANCE.NS has a trained agent"
)

# Dataset selector
dataset = st.sidebar.selectbox(
    "Select Dataset",
    ["Test Set (2022-2023)", "Validation Set (2021)", "Training Set (2018-2020)"]
)

# Strategy selector
strategies = st.sidebar.multiselect(
    "Compare Strategies",
    ["Trained Agent", "Buy-and-Hold", "Random", "Always Hold"],
    default=["Trained Agent", "Buy-and-Hold"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Multi-Stock Capability")
st.sidebar.info(
    """
    **Architecture supports any NSE stock!**
    
    Current: RELIANCE.NS (trained)
    
    To add new stocks:
    - Download data: `yfinance.download('TCS.NS')`
    - Train agent: `python scripts/run_training.py --ticker TCS.NS`
    - Load in dashboard: Ready to use!
    
    *Training time: ~30-60 min per stock*
    """
)

stock_input = st.sidebar.text_input(
    "Enter NSE ticker (e.g., TCS.NS)",
    value="RELIANCE.NS",
    disabled=True,
    help="Feature coming soon! Currently only RELIANCE.NS available"
)
# Add info box
st.sidebar.info(
    """
    **About This Project:**
    
    This is a Deep Reinforcement Learning agent that learns to trade stocks using:
    - DQN with target networks
    - Experience replay
    - Technical indicators (RSI, MACD, Bollinger Bands)
    - Risk-adjusted reward function
    
    Built with: PyTorch, OpenAI Gym, yfinance
    """
)

# Load data based on selection
@st.cache_data
def load_data(dataset_name):
    if dataset_name == "Test Set (2022-2023)":
        return pd.read_csv('data/splits/test.csv')
    elif dataset_name == "Validation Set (2021)":
        return pd.read_csv('data/splits/val.csv')
    else:
        return pd.read_csv('data/splits/train.csv')

# Run strategy
@st.cache_data
def run_strategy(data, strategy_name, _agent=None):
    """Run a trading strategy and return results"""
    env = TradingEnvironment(
        data=data,
        initial_balance=1000000,
        shares_per_trade=10,
        transaction_cost_pct=0.0001
    )
    
    state, _ = env.reset()
    portfolio_values = [env.portfolio_value]
    actions = []
    balances = [env.balance]
    shares = [env.shares_held]
    
    done = False
    step = 0
    
    while not done:
        if strategy_name == "Trained Agent":
            action = _agent.select_action(state, evaluation=True)
        elif strategy_name == "Buy-and-Hold":
            action = 1 if step == 0 else 0
        elif strategy_name == "Random":
            action = np.random.choice([0, 1, 2])
        else:  # Always Hold
            action = 0
        
        next_state, reward, done, truncated, info = env.step(action)
        
        actions.append(action)
        portfolio_values.append(info['portfolio_value'])
        balances.append(info['balance'])
        shares.append(info['shares_held'])
        
        state = next_state
        step += 1
    
    return {
        'portfolio_values': portfolio_values,
        'actions': actions,
        'balances': balances,
        'shares': shares,
        'final_value': portfolio_values[-1],
        'return': (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] * 100
    }

# Main content
try:
    # Load data
    data = load_data(dataset)
    st.success(f"Loaded {len(data)} trading days from {data['Date'].min()} to {data['Date'].max()}")
    
    # Load agent
    if "Trained Agent" in strategies:
        agent = DQNAgent(state_dim=13, action_dim=3)
        agent.load('models/checkpoints/best_agent.pth')
        st.success("Loaded trained DQN agent")
    else:
        agent = None
    
    # Run strategies
    with st.spinner("Running backtests..."):
        results = {}
        for strategy in strategies:
            results[strategy] = run_strategy(data, strategy, agent)
    
    # Metrics row
    st.header("Performance Metrics")
    cols = st.columns(len(strategies))
    
    for idx, strategy in enumerate(strategies):
        with cols[idx]:
            result = results[strategy]
            st.metric(
                label=strategy,
                value=f"₹{result['final_value']:,.0f}",
                delta=f"{result['return']:.2f}%"
            )
    
    # Portfolio value chart
    st.header("Portfolio Value Over Time")
    
    fig = go.Figure()
    
    for strategy in strategies:
        result = results[strategy]
        fig.add_trace(go.Scatter(
            x=list(range(len(result['portfolio_values']))),
            y=result['portfolio_values'],
            name=strategy,
            mode='lines',
            line=dict(width=2)
        ))
    
    fig.add_hline(y=1000000, line_dash="dash", line_color="gray", 
                  annotation_text="Initial Balance")
    
    fig.update_layout(
        xaxis_title="Trading Day",
        yaxis_title="Portfolio Value (₹)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Stock price for context
    st.header("Stock Price (RELIANCE)")
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=data.index[:len(results[strategies[0]]['portfolio_values'])],
        y=data['Close'][:len(results[strategies[0]]['portfolio_values'])],
        name='Stock Price',
        line=dict(color='black', width=2)
    ))
    
    fig2.update_layout(
        xaxis_title="Trading Day",
        yaxis_title="Price (₹)",
        height=400
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Actions visualization (for Trained Agent)
    if "Trained Agent" in strategies:
        st.header("Agent Actions")
        
        result = results["Trained Agent"]
        action_counts = {
            'HOLD': result['actions'].count(0),
            'BUY': result['actions'].count(1),
            'SELL': result['actions'].count(2)
        }
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Action Distribution")
            total = sum(action_counts.values())
            for action, count in action_counts.items():
                pct = count / total * 100
                st.metric(action, f"{count} times", f"{pct:.1f}%")
        
        with col2:
            # Plot actions on stock price
            fig3 = go.Figure()
            
            prices = data['Close'][:len(result['actions'])].values
            
            fig3.add_trace(go.Scatter(
                x=list(range(len(prices))),
                y=prices,
                name='Stock Price',
                line=dict(color='lightgray', width=1),
                mode='lines'
            ))
            
            # Add action markers
            for action_type, color, name in [(1, 'green', 'BUY'), (2, 'red', 'SELL')]:
                action_indices = [i for i, a in enumerate(result['actions']) if a == action_type]
                if action_indices:
                    fig3.add_trace(go.Scatter(
                        x=action_indices,
                        y=[prices[i] for i in action_indices],
                        name=name,
                        mode='markers',
                        marker=dict(size=10, color=color, symbol='triangle-up' if action_type == 1 else 'triangle-down')
                    ))
            
            fig3.update_layout(
                title="Trading Actions on Stock Price",
                xaxis_title="Trading Day",
                yaxis_title="Price (₹)",
                height=400
            )
            
            st.plotly_chart(fig3, use_container_width=True)
    
    # Comparison table
    st.header("Detailed Comparison")
    
    comparison_data = []
    for strategy in strategies:
        result = results[strategy]
        
        returns = np.diff(result['portfolio_values']) / result['portfolio_values'][:-1]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        peak = np.maximum.accumulate(result['portfolio_values'])
        drawdown = (np.array(result['portfolio_values']) - peak) / peak
        max_dd = np.min(drawdown) * 100
        
        comparison_data.append({
            'Strategy': strategy,
            'Final Value': f"₹{result['final_value']:,.0f}",
            'Return (%)': f"{result['return']:.2f}%",
            'Sharpe Ratio': f"{sharpe:.3f}",
            'Max Drawdown (%)': f"{max_dd:.2f}%"
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True)
    
    # Download results
    st.download_button(
        label="Download Results (CSV)",
        data=df_comparison.to_csv(index=False),
        file_name="backtest_results.csv",
        mime="text/csv"
    )

except FileNotFoundError as e:
    st.error(f"Error loading files: {e}")
    st.info("Make sure you've trained the agent and have data files in the correct locations.")
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.exception(e)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with Love using PyTorch, OpenAI Gym, and Streamlit</p>
        <p>Tech Stack: Python | PyTorch | Reinforcement Learning | Financial Data Analysis</p>
    </div>
    """,
    unsafe_allow_html=True
)