"""
Fetch historical Nifty 50 data for the last 15 years using NSE library.
Saves data in both CSV and Parquet formats for efficient storage and access.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from nsepy import get_history
import os
import pytz

def fetch_nifty_data(start_date, end_date):
    """
    Fetch Nifty 50 data from NSE for the given date range.
    
    Args:
        start_date: Start date for data fetch
        end_date: End date for data fetch
        
    Returns:
        pd.DataFrame: Nifty 50 data with OHLCV and additional columns
    """
    print(f"Fetching Nifty 50 data from {start_date} to {end_date}")
    
    try:
        # Fetch Nifty 50 data
        nifty_data = get_history(
            symbol="NIFTY 50",
            start=start_date,
            end=end_date,
            index=True
        )
        
        # Verify data is not empty
        if nifty_data.empty:
            raise ValueError("No data returned from NSE")
            
        print(f"Successfully fetched {len(nifty_data)} days of data")
        return nifty_data
        
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        raise

def process_data(df):
    """
    Process and clean the raw NSE data.
    
    Args:
        df: Raw DataFrame from NSE
        
    Returns:
        pd.DataFrame: Processed data with additional features
    """
    # Create copy to avoid modifying original
    processed = df.copy()
    
    # Ensure datetime index
    processed.index = pd.to_datetime(processed.index)
    
    # Calculate returns
    processed['Daily_Return'] = processed['Close'].pct_change()
    processed['Log_Return'] = np.log(processed['Close'] / processed['Close'].shift(1))
    
    # Calculate volatility (20-day rolling standard deviation of returns)
    processed['Volatility_20D'] = processed['Daily_Return'].rolling(window=20).std()
    
    # Calculate moving averages
    for period in [5, 10, 20, 50, 200]:
        processed[f'MA_{period}'] = processed['Close'].rolling(window=period).mean()
        
    # Add gap information
    processed['Gap'] = processed['Open'] - processed['Close'].shift(1)
    processed['Gap_Pct'] = (processed['Open'] - processed['Close'].shift(1)) / processed['Close'].shift(1)
    
    # Trading volume features
    processed['Volume_MA_5'] = processed['Volume'].rolling(window=5).mean()
    processed['Volume_MA_20'] = processed['Volume'].rolling(window=20).mean()
    processed['Relative_Volume'] = processed['Volume'] / processed['Volume_MA_20']
    
    # Drop rows with NaN values from calculations
    processed = processed.dropna()
    
    return processed

def main():
    # Create data directory if it doesn't exist
    data_dir = os.path.join('data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Calculate date range (15 years from today)
    end_date = datetime.now(pytz.timezone('Asia/Kolkata')).date()
    start_date = end_date - timedelta(days=15*365)  # Approximate 15 years
    
    try:
        # Fetch data
        nifty_data = fetch_nifty_data(start_date, end_date)
        
        # Process data
        processed_data = process_data(nifty_data)
        
        # Save raw data
        raw_csv_path = os.path.join(data_dir, 'nifty50_raw.csv')
        raw_parquet_path = os.path.join(data_dir, 'nifty50_raw.parquet')
        nifty_data.to_csv(raw_csv_path)
        nifty_data.to_parquet(raw_parquet_path)
        print(f"Raw data saved to {raw_csv_path} and {raw_parquet_path}")
        
        # Save processed data
        processed_csv_path = os.path.join(data_dir, 'nifty50_processed.csv')
        processed_parquet_path = os.path.join(data_dir, 'nifty50_processed.parquet')
        processed_data.to_csv(processed_csv_path)
        processed_data.to_parquet(processed_parquet_path)
        print(f"Processed data saved to {processed_csv_path} and {processed_parquet_path}")
        
        # Print data summary
        print("\nData Summary:")
        print(f"Date Range: {processed_data.index.min()} to {processed_data.index.max()}")
        print(f"Total Trading Days: {len(processed_data)}")
        print("\nFeatures in processed data:")
        for col in processed_data.columns:
            print(f"- {col}")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    main() 