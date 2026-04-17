"""
Feature Engineering Module
Creates technical indicators and labels for FTSE 100 prediction
"""

import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Create features for financial time series prediction"""
    
    def __init__(self, data):
        """
        Initialize feature engineer
        
        Args:
            data: pd.DataFrame with OHLCV data
        """
        self.data = data.copy()
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        
    def create_labels(self, horizon=1, method='binary'):
        """
        Create prediction labels
        
        Args:
            horizon: Number of days ahead to predict (default: 1)
            method: 'binary' for up/down, 'ternary' for up/flat/down
            
        Returns:
            pd.Series: Labels
        """
        print(f"\nCreating labels (horizon={horizon}, method={method})...")
        
        # Calculate future returns
        future_close = self.data['Close'].shift(-horizon)
        current_close = self.data['Close']
        future_return = (future_close - current_close) / current_close
        
        if method == 'binary':
            # Binary: 1 if up, 0 if down
            labels = (future_return > 0).astype(int)
            print(f"Label distribution: Up={labels.sum()}, Down={(~labels.astype(bool)).sum()}")
            
        elif method == 'ternary':
            # Ternary: 1 if up >0.5%, 0 if flat, -1 if down <-0.5%
            labels = pd.Series(0, index=self.data.index)
            labels[future_return > 0.005] = 1
            labels[future_return < -0.005] = -1
            print(f"Label distribution: Up={sum(labels==1)}, Flat={sum(labels==0)}, Down={sum(labels==-1)}")
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return labels
    
    def create_price_features(self):
        """Create price-based features"""
        print("\nCreating price features...")
        
        features = pd.DataFrame(index=self.data.index)
        
        # Returns
        features['return_1d'] = self.data['Close'].pct_change(1)
        features['return_5d'] = self.data['Close'].pct_change(5)
        features['return_20d'] = self.data['Close'].pct_change(20)
        
        # Log returns
        features['log_return_1d'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        
        # Price relative to moving averages
        features['price_sma_5'] = self.data['Close'] / self.data['Close'].rolling(5).mean()
        features['price_sma_20'] = self.data['Close'] / self.data['Close'].rolling(20).mean()
        features['price_sma_50'] = self.data['Close'] / self.data['Close'].rolling(50).mean()
        
        # High-Low spread
        features['hl_spread'] = (self.data['High'] - self.data['Low']) / self.data['Close']
        
        # Close relative to High-Low range
        features['close_position'] = (self.data['Close'] - self.data['Low']) / (self.data['High'] - self.data['Low'])
        
        return features
    
    def create_technical_indicators(self):
        """Create technical indicators"""
        print("Creating technical indicators...")
        
        features = pd.DataFrame(index=self.data.index)
        
        # Simple Moving Averages
        for window in [5, 10, 20, 50]:
            features[f'sma_{window}'] = self.data['Close'].rolling(window).mean()
        
        # Exponential Moving Averages
        for window in [5, 20]:
            features[f'ema_{window}'] = self.data['Close'].ewm(span=window, adjust=False).mean()
        
        # Moving Average Convergence Divergence (MACD)
        ema_12 = self.data['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = self.data['Close'].ewm(span=26, adjust=False).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_diff'] = features['macd'] - features['macd_signal']
        
        # Relative Strength Index (RSI)
        for window in [14, 28]:
            delta = self.data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            features[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        for window in [20]:
            sma = self.data['Close'].rolling(window).mean()
            std = self.data['Close'].rolling(window).std()
            features[f'bb_upper_{window}'] = sma + (2 * std)
            features[f'bb_lower_{window}'] = sma - (2 * std)
            features[f'bb_width_{window}'] = (features[f'bb_upper_{window}'] - features[f'bb_lower_{window}']) / sma
            features[f'bb_position_{window}'] = (self.data['Close'] - features[f'bb_lower_{window}']) / (features[f'bb_upper_{window}'] - features[f'bb_lower_{window}'])
        
        # Stochastic Oscillator
        window = 14
        low_min = self.data['Low'].rolling(window).min()
        high_max = self.data['High'].rolling(window).max()
        features['stoch_k'] = 100 * (self.data['Close'] - low_min) / (high_max - low_min)
        features['stoch_d'] = features['stoch_k'].rolling(3).mean()
        
        # Average True Range (ATR)
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        features['atr_14'] = true_range.rolling(14).mean()
        
        return features
    
    def create_volume_features(self):
        """Create volume-based features"""
        print("Creating volume features...")
        
        features = pd.DataFrame(index=self.data.index)
        
        # Volume changes
        features['volume_change_1d'] = self.data['Volume'].pct_change(1)
        features['volume_change_5d'] = self.data['Volume'].pct_change(5)
        
        # Volume relative to moving average
        features['volume_sma_5'] = self.data['Volume'] / self.data['Volume'].rolling(5).mean()
        features['volume_sma_20'] = self.data['Volume'] / self.data['Volume'].rolling(20).mean()
        
        # On-Balance Volume (OBV)
        obv = np.where(self.data['Close'] > self.data['Close'].shift(1), 
                       self.data['Volume'], 
                       np.where(self.data['Close'] < self.data['Close'].shift(1), 
                               -self.data['Volume'], 0))
        features['obv'] = pd.Series(obv).cumsum()
        
        return features
    
    def create_volatility_features(self):
        """Create volatility features"""
        print("Creating volatility features...")
        
        features = pd.DataFrame(index=self.data.index)
        
        # Historical volatility
        returns = np.log(self.data['Close'] / self.data['Close'].shift(1))
        for window in [5, 20, 50]:
            features[f'volatility_{window}'] = returns.rolling(window).std() * np.sqrt(252)
        
        # Parkinson volatility (using High-Low)
        for window in [20]:
            hl_ratio = np.log(self.data['High'] / self.data['Low'])
            features[f'parkinson_vol_{window}'] = (hl_ratio ** 2).rolling(window).mean() * np.sqrt(252 / (4 * np.log(2)))
        
        return features
    
    def create_momentum_features(self):
        """Create momentum features"""
        print("Creating momentum features...")
        
        features = pd.DataFrame(index=self.data.index)
        
        # Rate of Change (ROC)
        for window in [5, 10, 20]:
            features[f'roc_{window}'] = ((self.data['Close'] - self.data['Close'].shift(window)) / 
                                         self.data['Close'].shift(window)) * 100
        
        # Momentum
        for window in [5, 10]:
            features[f'momentum_{window}'] = self.data['Close'] - self.data['Close'].shift(window)
        
        return features
    
    def create_all_features(self, horizon=1, label_method='binary'):
        """
        Create all features and labels
        
        Args:
            horizon: Prediction horizon in days
            label_method: Label creation method
            
        Returns:
            pd.DataFrame: Complete feature set with labels
        """
        print("="*60)
        print("FEATURE ENGINEERING")
        print("="*60)
        
        # Create all feature sets
        price_features = self.create_price_features()
        technical_features = self.create_technical_indicators()
        volume_features = self.create_volume_features()
        volatility_features = self.create_volatility_features()
        momentum_features = self.create_momentum_features()
        
        # Combine all features
        all_features = pd.concat([
            self.data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']],
            price_features,
            technical_features,
            volume_features,
            volatility_features,
            momentum_features
        ], axis=1)
        
        # Create labels
        all_features['label'] = self.create_labels(horizon=horizon, method=label_method)
        
        # Remove rows with NaN (from rolling windows and future labels)
        initial_rows = len(all_features)
        all_features = all_features.dropna()
        final_rows = len(all_features)
        
        print(f"\nFeature creation complete:")
        print(f"Total features: {len(all_features.columns) - 7}")  # Exclude Date, OHLCV, label
        print(f"Rows removed (NaN): {initial_rows - final_rows}")
        print(f"Final dataset size: {final_rows} rows")
        print(f"\nFeature columns: {list(all_features.columns[6:-1])}")  # Exclude Date, OHLCV, label
        
        return all_features


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Create features from FTSE 100 data"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV file with raw data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory (default: data/processed)"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Prediction horizon in days (default: 1)"
    )
    parser.add_argument(
        "--label-method",
        type=str,
        default="binary",
        choices=["binary", "ternary"],
        help="Label creation method (default: binary)"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    data = pd.read_csv(args.input)
    
    # Create features
    engineer = FeatureEngineer(data)
    features = engineer.create_all_features(
        horizon=args.horizon,
        label_method=args.label_method
    )
    
    # Save features
    Path(args.output).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(args.output, f"features_h{args.horizon}_{args.label_method}.csv")
    features.to_csv(output_file, index=False)
    
    print("\n" + "="*60)
    print(f"Features saved to: {output_file}")
    print(f"Shape: {features.shape}")
    print("="*60)


if __name__ == "__main__":
    main()
