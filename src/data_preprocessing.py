import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy.stats import zscore

INPUT_DIR = Path("data/nifty50")
OUTPUT_DIR = Path("data/processed_nifty50_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def resample_weekly(df):
    df.set_index("Date", inplace=True)
    df = df.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
        "Ticker": "first"
        # "Sector": "first"  # Only include if available
    }).dropna().reset_index()
    df = df[df["Date"] >= pd.to_datetime("2004-04-01")].reset_index(drop=True)
    return df

def compute_log_returns(df):
    df["log_return"] = np.where(
        df["Close"].shift(1) > 0,
        np.log(df["Close"] / df["Close"].shift(1)),
        np.nan
    )
    return df

def compute_rolling_std(df, window=4):
    df["rolling_std"] = df["log_return"].rolling(window).std()
    return df

def compute_atr(df, window=4):
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window).mean()
    return df

def compute_volume_price_corr(df, window=4):
    df["vol_price_corr"] = df["Volume"].rolling(window).corr(df["Close"])
    return df

def compute_rsi(df, window=14):
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def compute_streaks(df):
    direction = np.sign(df["Close"].diff())
    streak_ids = (direction != direction.shift()).cumsum()
    df["streak"] = direction.groupby(streak_ids).cumsum().fillna(0)
    return df

def load_and_clean_nifty50_csv(file_path):
    # Read first three rows separately
    raw = pd.read_csv(file_path, header=None)

    # Extract ticker from the second row (row index 1)
    ticker = raw.iloc[1, 1]

    # Manually set column names
    columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
    
    # Drop the first 3 rows and reset the index
    df = raw.iloc[3:].copy()
    df.columns = columns
    df.reset_index(drop=True, inplace=True)

    # Convert date and volume
    df["Date"] = pd.to_datetime(df["Date"])
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    # Convert prices to float
    for col in ["Close", "High", "Low", "Open"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Add ticker column
    df["Ticker"] = ticker
    print(f"{ticker} | Start Date: {df['Date'].iloc[0].date()} | Rows: {len(df)}")
    return df

def process_nifty50_data():
    for file in INPUT_DIR.glob("*.csv"):
        df = load_and_clean_nifty50_csv(file)

        df = resample_weekly(df)
        df["Wick_length"] = df["High"] - df[["Open", "Close"]].max(axis=1)
        df["Shadow_length"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
        df["Body_length"] = (df["Open"] - df["Close"])

        # Feature engineering
        df = compute_log_returns(df)
        df = compute_rolling_std(df)
        df = compute_atr(df)
        df = compute_volume_price_corr(df)
        df = compute_rsi(df)
        df = compute_streaks(df)

        #To remove rows with null technical indicator values
        df = df.iloc[14:].reset_index(drop=True)

        df.to_csv(OUTPUT_DIR / file.name, index=False)

if __name__ == "__main__":
    process_nifty50_data()