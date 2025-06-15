import os
import pandas as pd
import numpy as np
from typing import List, Tuple
from tqdm import tqdm

def load_data(data_folder: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    all_data = []
    tickers = []

    for file in tqdm(csv_files, desc="Loading CSVs"):
        ticker = file.replace('.csv', '')
        tickers.append(ticker)
        df = pd.read_csv(os.path.join(data_folder, file), parse_dates=['Date'])
        df['Ticker'] = ticker
        all_data.append(df)

    df_all = pd.concat(all_data)
    df_all = df_all.sort_values(['Date', 'Ticker']).reset_index(drop=True)

    feature_cols = [col for col in df_all.columns if col not in ['Date', 'Ticker']]
    df_pivot = df_all.pivot(index='Date', columns='Ticker', values=feature_cols)
    df_pivot = df_pivot.sort_index()

    # Flatten MultiIndex columns: (feature, ticker) → "ticker__feature"
    df_pivot.columns = [f"{ticker}__{feat}" for feat, ticker in df_pivot.columns]

    return df_pivot, sorted(tickers), feature_cols

def make_sequences_targets(
    df: pd.DataFrame,
    tickers: List[str],
    feature_cols: List[str],
    sequence_length: int = 12,
    pred_1m: int = 4,
    pred_2m: int = 8,
    target_feature: str = "log_return_Close"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if target_feature not in feature_cols:
        raise ValueError(f"Target feature '{target_feature}' not found in features")

    num_stocks = len(tickers)
    num_features = len(feature_cols)

    # 3D array: [time, stocks, features]
    data = np.full((len(df), num_stocks, num_features), np.nan)

    for i, ticker in enumerate(tickers):
        for j, feat in enumerate(feature_cols):
            col = f"{ticker}__{feat}"
            if col in df.columns:
                data[:, i, j] = df[col].values

    total_samples = len(df) - sequence_length - pred_2m
    X, Y, mask = [], [], []

    for t in tqdm(range(total_samples), desc="Creating sequences"):
        seq = data[t : t + sequence_length]
        seq = np.nan_to_num(seq, nan=0.0)  # Fill NaNs in input

        target_1m = data[t + sequence_length + pred_1m - 1, :, feature_cols.index(target_feature)]
        target_2m = data[t + sequence_length + pred_2m - 1, :, feature_cols.index(target_feature)]

        valid_mask = ~(np.isnan(target_1m) | np.isnan(target_2m))
        target_1m = np.nan_to_num(target_1m, nan=0.0)
        target_2m = np.nan_to_num(target_2m, nan=0.0)

        X.append(seq)
        Y.append(np.stack([target_1m, target_2m], axis=-1))  # [stocks, 2]
        mask.append(valid_mask)

    return (
        np.array(X),    # [N, seq_len, stocks, features]
        np.array(Y),    # [N, stocks, 2]
        np.array(mask)  # [N, stocks]
    )

def train_test_split(
    X: np.ndarray, Y: np.ndarray, mask: np.ndarray,
    test_ratio: float = 0.2, drop_last_weeks: int = 12
) -> Tuple[dict, dict]:
    num_samples = X.shape[0]
    split_point = int(num_samples * (1 - test_ratio))

    if drop_last_weeks >= (num_samples - split_point):
        raise ValueError("drop_last_weeks is too large, resulting in an empty test set.")

    train = {
        "X": X[:split_point],
        "Y": Y[:split_point],
        "mask": mask[:split_point],
    }

    test = {
        "X": X[split_point:-drop_last_weeks],
        "Y": Y[split_point:-drop_last_weeks],
        "mask": mask[split_point:-drop_last_weeks],
    }

    print(f"Train: {train['X'].shape[0]} samples | Test: {test['X'].shape[0]} samples")
    return train, test

def save_data(train: dict, test: dict, tickers: List[str], features: List[str], output_path: str, save_sample_csv: bool = True):
    os.makedirs(output_path, exist_ok=True)

    np.save(os.path.join(output_path, "train_X.npy"), train["X"])
    np.save(os.path.join(output_path, "train_Y.npy"), train["Y"])
    np.save(os.path.join(output_path, "train_mask.npy"), train["mask"])

    np.save(os.path.join(output_path, "test_X.npy"), test["X"])
    np.save(os.path.join(output_path, "test_Y.npy"), test["Y"])
    np.save(os.path.join(output_path, "test_mask.npy"), test["mask"])

    np.save(os.path.join(output_path, "tickers.npy"), np.array(tickers))
    np.save(os.path.join(output_path, "features.npy"), np.array(features))

    print("Saved all data to:", output_path)

    # Optional: save one input sequence and target to CSV for debugging
    if save_sample_csv:
        sample_idx = 0
        X_sample = train["X"][sample_idx]  # [seq_len, stocks, features]
        Y_sample = train["Y"][sample_idx]  # [stocks, 2]

        seq_len, num_stocks, num_feats = X_sample.shape
        for i in range(num_stocks):
            stock_data = pd.DataFrame(X_sample[:, i, :], columns=features)
            stock_data.to_csv(os.path.join(output_path, f"sample_seq_{tickers[i]}.csv"), index=False)
        pd.DataFrame(Y_sample, columns=["target_1m", "target_2m"]).to_csv(
            os.path.join(output_path, "sample_targets.csv"), index=False
        )

def prepare_data(data_folder: str, output_folder: str):
    df_pivot, tickers, feature_cols = load_data(data_folder)
    X, Y, mask = make_sequences_targets(df_pivot, tickers, feature_cols)
    train, test = train_test_split(X, Y, mask)
    save_data(train, test, tickers, feature_cols, output_folder)

if __name__ == "__main__":
    prepare_data(
        data_folder=r"data\processed_nifty50_data",
        output_folder=r"data\train_test"
    )