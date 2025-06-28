import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

INPUT_DIR = Path("data/nifty50")
OUTPUT_DIR = Path("data/processed_nifty50_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPSILON = 1e-10

def compute_log_returns_all(df, epsilon=1e-8):
    """
    Compute log returns with proper NaN handling - no row dropping
    Formula: log_return = ln(price_t / price_{t-1})
    """
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        curr = df[col].astype(float).copy()
        prev = df[col].shift(1).astype(float).copy()
        
        valid_mask = (curr > 0) & (prev > 0) & (curr.notna()) & (prev.notna())
        log_returns = pd.Series(0.0, index=df.index)
        log_returns.loc[valid_mask] = np.log(curr.loc[valid_mask] / prev.loc[valid_mask])
        log_returns.replace([np.inf, -np.inf], 0.0, inplace=True)
        
        df[f"log_return_{col}"] = log_returns
    
    df.drop(columns=["Open", "High", "Low", "Close", "Volume"], inplace=True)
    return df

def compute_features(df):
    """
    Features at multiple time horizons
    """
    # Standard volatility measures
    df["volatility_20d"] = (df["log_return_Close"].rolling(20, min_periods=1).std()).fillna(0)
    df["volatility_60d"] = (df["log_return_Close"].rolling(60, min_periods=1).std()).fillna(0)
    
    # Volatility of volatility (uncertainty measure)
    # Formula: vol_of_vol = std(volatility_20d)
    df["vol_of_vol_60d"] = df["volatility_20d"].rolling(60, min_periods=1).std().fillna(0)

    # Multi-scale momentum
    df["momentum_10d"] = (df["log_return_Close"].rolling(10, min_periods=1).sum()).fillna(0)
    df["momentum_20d"] = (df["log_return_Close"].rolling(20, min_periods=1).sum()).fillna(0)
    df["momentum_60d"] = (df["log_return_Close"].rolling(60, min_periods=1).sum()).fillna(0)
    
    # Momentum acceleration (change in momentum)
    # Formula: momentum_accel = momentum_t - momentum_{t-10}
    df["momentum_accel_20d"] = df["momentum_20d"] - df["momentum_20d"].shift(10)
    df["momentum_accel_20d"] = df["momentum_accel_20d"].fillna(0)

    # Kurtosis (tail heaviness measure)
    # Formula: kurtosis = E[(X - μ)⁴] / σ⁴ - 3
    df["kurtosis_20d"] = df["log_return_Close"].rolling(20, min_periods=5).kurt().fillna(0)
    df["kurtosis_60d"] = df["log_return_Close"].rolling(60, min_periods=10).kurt().fillna(0)
    
    # Downside vs upside volatility
    returns_20d = df["log_return_Close"].rolling(20, min_periods=1)
    returns_60d = df["log_return_Close"].rolling(60, min_periods=1)
    
    df["downside_vol_20d"] = returns_20d.apply(lambda x: x[x < 0].std() if len(x[x < 0]) > 0 else 0).fillna(0)
    df["upside_vol_20d"] = returns_20d.apply(lambda x: x[x > 0].std() if len(x[x > 0]) > 0 else 0).fillna(0)
    df["downside_vol_60d"] = returns_60d.apply(lambda x: x[x < 0].std() if len(x[x < 0]) > 0 else 0).fillna(0)
    df["upside_vol_60d"] = returns_60d.apply(lambda x: x[x > 0].std() if len(x[x > 0]) > 0 else 0).fillna(0)
    
    # Volatility asymmetry
    # Formula: vol_asymmetry = (upside_vol - downside_vol) / (upside_vol + downside_vol)
    df["vol_asymmetry_20d"] = (df["upside_vol_20d"] - df["downside_vol_20d"]) / (df["upside_vol_20d"] + df["downside_vol_20d"] + EPSILON)
    df["vol_asymmetry_60d"] = (df["upside_vol_60d"] - df["downside_vol_60d"]) / (df["upside_vol_60d"] + df["downside_vol_60d"] + EPSILON)
    
    # Multi-scale volume patterns
    df["volume_ma_20d"] = df["log_return_Volume"].rolling(20, min_periods=1).mean().fillna(0)
    df["volume_ma_60d"] = df["log_return_Volume"].rolling(60, min_periods=1).mean().fillna(0)
    
    # Volume-price correlations
    # Formula: correlation coefficient between volume and price changes
    df["vol_price_corr_60d"] = df["log_return_Volume"].rolling(60, min_periods=1).corr(
        df["log_return_Close"]
    ).fillna(0)

    # Distance from moving averages (mean reversion signals)
    ma_60d = df["log_return_Close"].rolling(60, min_periods=1).mean().fillna(0)
    df["distance_from_ma_60d"] = df["log_return_Close"] - ma_60d
    
    # Bollinger Band position
    # Formula: bb_position = (price - MA) / (2 * std)
    std_60d = df["log_return_Close"].rolling(60, min_periods=1).std().fillna(0)
    df["bb_position_60d"] = ((df["log_return_Close"] - ma_60d) / (2 * std_60d + EPSILON)).fillna(0)
    
    # Patch-level statistics
    df["patch_mean_10d"] = df["log_return_Close"].rolling(10, min_periods=1).mean().fillna(0)
    df["patch_std_10d"] = df["log_return_Close"].rolling(10, min_periods=1).std().fillna(0)
    
    return df

def load_and_clean_nifty50_csv(file_path):
    """Load and clean Nifty50 CSV data"""
    try:
        raw = pd.read_csv(file_path, header=None)
        ticker = raw.iloc[1, 1]
        
        # Define columns
        columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
        df = raw.iloc[3:].copy()
        df.columns = columns
        df.reset_index(drop=True, inplace=True)
        
        # Convert data types
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        
        # Convert numeric columns
        for col in ["Close", "High", "Low", "Open", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Add ticker
        df["Ticker"] = ticker
        
        # Filter date range
        df = df[df["Date"] >= pd.to_datetime("2004-04-01")].reset_index(drop=True)
        
        # Only remove rows where ALL OHLCV values are NaN (completely empty rows)
        ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
        all_nan_mask = df[ohlcv_cols].isna().all(axis=1)
        if all_nan_mask.any():
            print(f"Warning: {file_path.name} removing {all_nan_mask.sum()} completely empty rows")
            df = df[~all_nan_mask].reset_index(drop=True)
        
        # Basic data validation - updated minimum requirement
        if len(df) < 120:  # Still need sufficient data for processing
            print(f"Warning: {file_path.name} has insufficient data ({len(df)} rows)")
            return None
            
        return df
        
    except Exception as e:
        print(f"Error processing {file_path.name}: {str(e)}")
        return None

def process_nifty50_data():
    """Main processing function"""
    processed_files = 0
    failed_files = []
    
    for file in INPUT_DIR.glob("*.csv"):
        try:
            print(f"Processing {file.name}...")
            df = load_and_clean_nifty50_csv(file)
            
            if df is None:
                failed_files.append(file.name)
                continue
            
            # Compute candlestick features with NaN handling
            wick_length = df["High"] - df[["Open", "Close"]].max(axis=1)
            shadow_length = df[["Open", "Close"]].min(axis=1) - df["Low"]
            body_length = abs(df["Open"] - df["Close"])
            
            df["Wick_length"] = wick_length.fillna(0)
            df["Shadow_length"] = shadow_length.fillna(0)
            df["Body_length"] = body_length.fillna(0)
            
            # Handle illiquid rows (all OHLC values are the same)
            illiquid_mask = (
                (df["Open"] == df["High"]) &
                (df["Open"] == df["Low"]) &
                (df["Open"] == df["Close"])
            )
            
            # Set candlestick features to 0 for illiquid rows
            for col in ["Wick_length", "Shadow_length", "Body_length"]:
                df.loc[illiquid_mask, col] = 0.0
            
            # Compute log returns
            df = compute_log_returns_all(df)
            
            # Set log returns to 0 for illiquid rows (preserve date sequence)
            log_return_cols = ["log_return_Open", "log_return_High", "log_return_Low", "log_return_Close", "log_return_Volume"]
            for col in log_return_cols:
                if col in df.columns:
                    df.loc[illiquid_mask, col] = 0.0
            
            # Compute all technical indicators in organized order
            df = compute_features(df)
            
            # Drop the first 60 rows as they won't have sufficient data
            df = df.iloc[60:].reset_index(drop=True)
            
            # Final data quality check
            if len(df) == 0:
                print(f"Warning: {file.name} has no data after processing and dropping first 60 rows")
                failed_files.append(file.name)
                continue
            
            # Save processed data
            output_file = OUTPUT_DIR / file.name
            df.to_csv(output_file, index=False)
            processed_files += 1
            
            # Print some statistics
            total_rows = len(df)
            non_zero_features = sum((df.select_dtypes(include=[np.number]) != 0).any())
            print(f"  - Processed {total_rows} rows (dropped first 60), {non_zero_features} features with non-zero values")
            
        except Exception as e:
            print(f"Error processing {file.name}: {str(e)}")
            failed_files.append(file.name)
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_files} files")
    if failed_files:
        print(f"Failed files: {failed_files}")

if __name__ == "__main__":
    print("Started preprocessing")
    process_nifty50_data()
    print("Preprocessing complete!")