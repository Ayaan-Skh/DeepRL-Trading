import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
plt.rcParams['figure.figsize']=(15,10)

df=pd.read_csv("data/processed/reliance_processed.csv")
df['Date']=pd.to_datetime(df['Date']) # Convert 'Date' column to datetime

#  Create subplots
fig, axes = plt.subplots(4, 1, figsize=(15, 12))

# 1. Price and Moving Averages
axes[0].plot(df['Date'], df['Close'], label='Close Price', linewidth=1)
axes[0].plot(df['Date'], df['SMA_20'], label='SMA 20', linewidth=1, linestyle='--')
axes[0].plot(df['Date'], df['SMA_50'], label='SMA 50', linewidth=1, linestyle='--')
axes[0].set_title('Stock Price and Moving Averages', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Price (₹)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)


# 2. RSI
axes[1].plot(df['Date'], df['RSI_14'], label='RSI (14)', color='purple', linewidth=1)
axes[1].axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
axes[1].axhline(y=30, color='g', linestyle='--', label='Oversold (30)')
axes[1].fill_between(df['Date'], 30, 70, alpha=0.1, color='gray')
axes[1].set_title('Relative Strength Index (RSI)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('RSI')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. MACD
axes[2].plot(df['Date'], df['MACD'], label='MACD', linewidth=1)
axes[2].plot(df['Date'], df['MACD_Signal'], label='Signal', linewidth=1, linestyle='--')
axes[2].bar(df['Date'], df['MACD'] - df['MACD_Signal'], 
            label='Histogram', alpha=0.3, width=1)
axes[2].set_title('MACD (Moving Average Convergence Divergence)', fontsize=14, fontweight='bold')
axes[2].set_ylabel('MACD')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# 4. Bollinger Bands
axes[3].plot(df['Date'], df['Close'], label='Close Price', linewidth=1, color='blue')
axes[3].plot(df['Date'], df['BB_Upper'], label='Upper Band', 
             linewidth=1, linestyle='--', color='red', alpha=0.7)
axes[3].plot(df['Date'], df['BB_Middle'], label='Middle Band (SMA 20)', 
             linewidth=1, linestyle='--', color='orange', alpha=0.7)
axes[3].plot(df['Date'], df['BB_Lower'], label='Lower Band', 
             linewidth=1, linestyle='--', color='green', alpha=0.7)
axes[3].fill_between(df['Date'], df['BB_Lower'], df['BB_Upper'], alpha=0.1, color='gray')
axes[3].set_title('Bollinger Bands', fontsize=14, fontweight='bold')
axes[3].set_ylabel('Price (₹)')
axes[3].set_xlabel('Date')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/plots/technical_indicators.png', dpi=300, bbox_inches='tight')
print("Saved visualization to results/plots/technical_indicators.png")
plt.show()