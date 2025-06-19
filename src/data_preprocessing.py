import pandas as pd
import numpy as np
from pathlib import Path

INPUT_DIR = Path("data/nifty50")
OUTPUT_DIR = Path("data/processed_nifty50_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPSILON = 1e-10

def compute_log_returns_all(df, epsilon=1e-8):
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        curr = df[col].astype(float).copy()
        prev = df[col].shift(1).astype(float).copy()
        prev[prev < epsilon] = epsilon
        curr[curr < epsilon] = epsilon
        df[f"log_return_{col}"] = np.log(curr / prev)
    df.drop(columns=["Open", "High", "Low", "Close", "Volume"], inplace=True)
    return df

def compute_rolling_std(df, window=4):
    df["rolling_std"] = df["log_return_Close"].rolling(window).std()
    return df

def compute_atr(df, window=4):
    exp_H = np.exp(df["log_return_High"])
    exp_L = np.exp(df["log_return_Low"])
    exp_C = np.exp(df["log_return_Close"])
    exp_C_shift = np.exp(df["log_return_Close"].shift())

    high_low = exp_H - exp_L
    high_close = np.abs(exp_H - exp_C_shift)
    low_close = np.abs(exp_L - exp_C_shift)

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window).mean()
    return df

def compute_volume_price_corr(df, window=4):
    df["vol_price_corr"] = df["log_return_Volume"].rolling(window).corr(df["log_return_Close"])
    return df

def compute_rsi(df, window=14):
    delta = df["log_return_Close"].fillna(0)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean().replace(0, EPSILON)

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].clip(0, 100)
    return df

def compute_streaks(df):
    direction = np.sign(df["log_return_Close"].fillna(0))
    streak_ids = (direction != direction.shift()).cumsum()
    df["streak"] = direction.groupby(streak_ids).cumsum()
    return df

def compute_macd(df, short=12, long=26, signal=9):
    ema_short = df["log_return_Close"].ewm(span=short, adjust=False).mean()
    ema_long = df["log_return_Close"].ewm(span=long, adjust=False).mean().replace(0, EPSILON)
    df["MACD"] = ema_short - ema_long
    df["MACD_signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    return df

def compute_moving_averages(df):
    df["MA_quarter"] = df["log_return_Close"].rolling(window=13).mean()
    return df

def load_and_clean_nifty50_csv(file_path):
    raw = pd.read_csv(file_path, header=None)
    ticker = raw.iloc[1, 1]
    columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
    df = raw.iloc[3:].copy()
    df.columns = columns
    df.reset_index(drop=True, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    for col in ["Close", "High", "Low", "Open"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Ticker"] = ticker
    df = df[df["Date"] >= pd.to_datetime("2004-04-01")].reset_index(drop=True)
    return df

def process_nifty50_data():
    for file in INPUT_DIR.glob("*.csv"):
        df = load_and_clean_nifty50_csv(file)

        df["Wick_length"] = df["High"] - df[["Open", "Close"]].max(axis=1)
        df["Shadow_length"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
        df["Body_length"] = df["Open"] - df["Close"]

        # Detect illiquid rows
        illiquid_mask = (
            (df["Open"] == df["High"]) &
            (df["Open"] == df["Low"]) &
            (df["Open"] == df["Close"])
        )

        # Set zero for log returns and candle features on illiquid rows
        for col in ["Wick_length", "Shadow_length", "Body_length"]:
            df.loc[illiquid_mask, col] = 0.0

        df = compute_log_returns_all(df)
        for col in ["log_return_Open", "log_return_High", "log_return_Low", "log_return_Close", "log_return_Volume"]:
            df.loc[illiquid_mask, col] = 0.0

        df = compute_rolling_std(df)
        df = compute_atr(df)
        df = compute_volume_price_corr(df)
        df = compute_rsi(df)
        df = compute_streaks(df)
        df = compute_macd(df)
        df = compute_moving_averages(df)

        df = df.iloc[14:].reset_index(drop=True)
        df.to_csv(OUTPUT_DIR / file.name, index=False)

if __name__ == "__main__":
    process_nifty50_data()