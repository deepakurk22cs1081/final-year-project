"""
Data Collection Module for FTSE 100 Index
Downloads historical FTSE 100 data and saves to CSV
"""

import argparse
import os
from datetime import datetime
import pandas as pd
import yfinance as yf
from pathlib import Path


class FTSEDataCollector:
    """Download and save FTSE 100 historical data"""
    
    def __init__(self, ticker="^FTSE", start_date="2010-01-01", end_date=None):
        """
        Initialize data collector
        
        Args:
            ticker: Yahoo Finance ticker symbol for FTSE 100 (default: ^FTSE)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (default: today)
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        
    def download_data(self):
        """
        Download FTSE 100 data from Yahoo Finance
        
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        print(f"Downloading {self.ticker} data from {self.start_date} to {self.end_date}...")
        
        try:
            data = yf.download(
                self.ticker,
                start=self.start_date,
                end=self.end_date,
                progress=True
            )
            
            if data.empty:
                raise ValueError("No data downloaded. Check ticker symbol and date range.")
            
            # Flatten multi-level columns produced by yfinance >= 0.2.40
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]
            
            # Reset index to make Date a column
            data.reset_index(inplace=True)
            
            print(f"Successfully downloaded {len(data)} rows of data")
            print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
            
            return data
            
        except Exception as e:
            print(f"Error downloading data: {e}")
            raise
    
    def validate_data(self, data):
        """
        Validate downloaded data
        
        Args:
            data: pd.DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: Validated data
        """
        print("\nValidating data...")
        
        # Check for missing values
        missing = data.isnull().sum()
        if missing.any():
            print(f"Warning: Missing values found:\n{missing[missing > 0]}")
        
        # Check for duplicate dates
        duplicates = data['Date'].duplicated().sum()
        if duplicates > 0:
            print(f"Warning: {duplicates} duplicate dates found. Removing...")
            data = data.drop_duplicates(subset='Date', keep='first')
        
        # Sort by date
        data = data.sort_values('Date').reset_index(drop=True)
        
        # Basic statistics
        print(f"\nData Statistics:")
        print(f"Number of trading days: {len(data)}")
        print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
        print(f"\nPrice Statistics:")
        print(data[['Open', 'High', 'Low', 'Close', 'Volume']].describe())
        
        return data
    
    def save_data(self, data, output_dir="data/raw"):
        """
        Save data to CSV
        
        Args:
            data: pd.DataFrame to save
            output_dir: Directory to save data
            
        Returns:
            str: Path to saved file
        """
        # Create directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate filename with date range
        start = data['Date'].min().strftime("%Y%m%d")
        end = data['Date'].max().strftime("%Y%m%d")
        filename = f"ftse100_{start}_{end}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # Save to CSV
        data.to_csv(filepath, index=False)
        print(f"\nData saved to: {filepath}")
        
        return filepath
    
    def run(self, output_dir="data/raw"):
        """
        Run complete data collection pipeline
        
        Args:
            output_dir: Directory to save data
            
        Returns:
            tuple: (pd.DataFrame, str) data and filepath
        """
        data = self.download_data()
        data = self.validate_data(data)
        filepath = self.save_data(data, output_dir)
        
        return data, filepath


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Download FTSE 100 historical data from Yahoo Finance"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="^FTSE",
        help="Yahoo Finance ticker symbol (default: ^FTSE)"
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2010-01-01",
        help="Start date in YYYY-MM-DD format (default: 2010-01-01)"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date in YYYY-MM-DD format (default: today)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="Output directory (default: data/raw)"
    )
    
    args = parser.parse_args()
    
    # Create collector and run
    collector = FTSEDataCollector(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end
    )
    
    data, filepath = collector.run(output_dir=args.output)
    
    print("\n" + "="*60)
    print("Data collection complete!")
    print(f"File: {filepath}")
    print(f"Rows: {len(data)}")
    print("="*60)


if __name__ == "__main__":
    main()
