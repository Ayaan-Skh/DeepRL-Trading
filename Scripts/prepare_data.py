import sys
sys.path.append('.')  # Add current directory to Python path

from src.data_loader import (
    download_stock_data,
    add_technical_indicators,
    create_train_val_test_splits
)

def main():
    # Configuration
    TICKER = "RELIANCE.NS"
    START_DATE = "2018-01-01"
    END_DATE = "2023-12-31"
    
    # Step 1: Download data
    raw_data = download_stock_data(
        ticker=TICKER,
        start_date=START_DATE,
        end_date=END_DATE,
        save_path="data/raw/reliance_raw.csv"
    )
    
    # Step 2: Add technical indicators
    processed_data = add_technical_indicators(raw_data)
    
    # Save processed data
    processed_data.to_csv("data/processed/reliance_processed.csv", index=False)
    print("Saved processed data")
    
    # Step 3: Create splits
    train_df, val_df, test_df = create_train_val_test_splits(processed_data)
    
    # Save splits
    train_df.to_csv("data/splits/train.csv", index=False)
    val_df.to_csv("data/splits/val.csv", index=False)
    test_df.to_csv("data/splits/test.csv", index=False)
    print("Saved train/val/test splits")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("DATA SUMMARY")
    print("="*50)
    print(f"Ticker: {TICKER}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Total trading days: {len(processed_data)}")
    print(f"\nPrice range:")
    print(f"  Min: ₹{processed_data['Close'].min():.2f}")
    print(f"  Max: ₹{processed_data['Close'].max():.2f}")
    print(f"  Mean: ₹{processed_data['Close'].mean():.2f}")
    print(f"\nTechnical Indicators:")
    print(f"  RSI range: {processed_data['RSI_14'].min():.1f} to {processed_data['RSI_14'].max():.1f}")
    print(f"  MACD range: {processed_data['MACD'].min():.2f} to {processed_data['MACD'].max():.2f}")

if __name__ == "__main__":
    main()