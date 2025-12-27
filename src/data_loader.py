import yfinance as yf
import pandas as pd
import ta
from pathlib import Path
import numpy as np
from src.logger import logging

def download_stock_data(ticker, start_date, end_date,save_path=None):
    """
    Download stock data from Yahoo Finance
    
    Args:
        ticker: Stock symbol (e.g., 'RELIANCE.NS')
        start_date: Start date (e.g., '2018-01-01')
        end_date: End date (e.g., '2023-12-31')
        save_path: Where to save CSV file (optional)
    
    Returns:
        DataFrame with OHLCV data
    """
    print(f"Downloading {ticker} data from {start_date} to {end_date}")
    df=yf.download(ticker,start=start_date,end=end_date)
    df.columns = df.columns.get_level_values(0)
    if df.empty:
        return ValueError(f"No data downloaded for {ticker}")
    df=df.reset_index() # Converts the dates to a column instead of keeping it as index
    
    #Keeping only essential columns
    df=df[['Date',"Open","High","Low","Close","Volume"]]
    
    df=df.dropna()
    if save_path:
        print(save_path)
        Path(save_path).parent.mkdir(parents=True,exist_ok=True)
        df.to_csv(save_path,index=False)
        print(f"Data saved to {save_path}")
    
    print(f"Downloaded {len(df)} days of data")
    return df    
    
def add_technical_indicators(df):
    # Add technical indicators to the OHLCV dataframe.
    # This function calculates and appends various technical indicators to a dataframe
    # containing Open, High, Low, Close, and Volume (OHLCV) data.
    # :param df: DataFrame with OHLCV data containing columns for Open, High, Low, Close, and Volume
    # :return: DataFrame with added technical indicator columns
    # :rtype: pandas.DataFrame
    """
    Docstring for add_technical_indicators
    
    :param df: Dataframe with OHLCV data 
    :return: Description
    :rtype: Any
    """
    print("computing technical indicators...")
    df=df.copy()
    
    # Calculate Simple Moving Average (SMA) for 20 past days
    df['SMA_20']=df['Close'].rolling(window=20).mean()
    df['SMA_50']=df['Close'].rolling(window=50).mean()
    print(df["Close"].shape)
    # Calculate Relative Strength Index (RSI) for past 14 days
    # 2. Relative Strength Index (RSI)
    df['RSI_14'] = ta.momentum.RSIIndicator(
        close=df['Close'], 
        window=14
    ).rsi()
    # ta.momentum.RSIIndicator: Creates RSI calculator
    # close=df['Close']: Use closing prices
    # window=14: Calculate RSI over 14 days (standard)
    # .rsi(): Actually compute the RSI values
    
    # For each day:
    # 1. Look at last 14 days
    # 2. Count up days (gains) and down days (losses)
    # 3. Compute: RSI = 100 - [100 / (1 + avg_gain/avg_loss)]
    
    # Calculate the MACD value for past 12(fast) and 26(slow) days
    macd_indicator=ta.trend.MACD(
        close=df['Close'],
        window_slow=26,  # Slow moving average (26 days)
        window_fast=12,  # Fast moving average (12 days)
        window_sign=9    # Signal line smoothing (9 days)
    )
    df['MACD']=macd_indicator.macd()
    df['MACD_Signal']=macd_indicator.macd_signal()
    # Creates MACD calculator with standard parameters (12, 26, 9)
    # .macd(): Gets MACD line (fast MA - slow MA)
    # .macd_signal(): Gets signal line (smoothed MACD)
     
    # 4. Bollinger Bands
    bollinger=ta.volatility.BollingerBands(
        close=df['Close'],
        window=20,
        window_dev=2
    )  
    df['BB_Upper'] = bollinger.bollinger_hband()
    df['BB_Lower'] = bollinger.bollinger_lband()
    df['BB_Middle'] = bollinger.bollinger_mavg()
    # What this does:

    # Calculates 20-day SMA (middle band)
    # Calculates standard deviation of last 20 days
    # Upper = Middle + 2×std
    # Lower = Middle - 2×std
    
    # 5. Volume Ratio
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
    # What this does:

    # Calculate average volume over 20 days
    # Divide today's volume by average
    # Result: 1.0 = normal, 2.0 = double normal, 0.5 = half normal
    
    # 6. Price Change Percentage
    df['Price_Change_Pct'] = df['Close'].pct_change() * 100
    # Drop rows with any NaN values resulting in technical indocator calculations
    df=df.dropna()
    
    print(f"Added technical indicators. {len(df)} rows remaining.")
    return df
    
def create_train_val_test_splits(df,train_ratio=0.7,val_ratio=0.15):
    """
    To split data chronologically in train, val, and test sets
    
        :param df: Dataframe with all data
        :param train_ratio: Train test split ratio
        :param val_ratio: Fraction of data for validation set
    Returns:
        train_df, val_df, test_df
    
    """
    df=df.sort_values('Date').reset_index(drop=True)
    n=len(df)
    train_end:int=n*train_ratio
    val_end:int=n*(train_ratio+val_ratio)
    # Split the data
    train_df=df.iloc[:int(train_end)].copy()
    val_df=df.iloc[int(train_end):int(val_end)].copy()
    test_df=df.iloc[int(val_end):].copy()

    print(f"\nData split:")
    print(f"Train: {len(train_df)} rows ({train_df['Date'].min()} to {train_df['Date'].max()})")
    print(f"Val:   {len(val_df)} rows ({val_df['Date'].min()} to {val_df['Date'].max()})")
    print(f"Test:  {len(test_df)} rows ({test_df['Date'].min()} to {test_df['Date'].max()})")
    
    return train_df, val_df, test_df


    
    