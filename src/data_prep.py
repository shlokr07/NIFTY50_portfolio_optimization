import os
import pandas as pd
import numpy as np
from typing import List, Tuple
from tqdm import tqdm

SEQ_LEN = 120
HORIZON1 = 20
HORIZON2 = 60

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

    df_pivot.columns = [f"{ticker}__{feat}" for feat, ticker in df_pivot.columns]

    return df_pivot, sorted(tickers), feature_cols


def make_sequences_targets(
    df: pd.DataFrame,
    tickers: List[str],
    feature_cols: List[str],
    sequence_length: int = SEQ_LEN,
    pred_steps: List[int] = [HORIZON1, HORIZON2],
    target_feature: str = "log_return_Close"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if target_feature not in feature_cols:
        raise ValueError(f"Target feature '{target_feature}' not found in features")

    num_stocks = len(tickers)
    num_features = len(feature_cols)

    data = np.full((len(df), num_stocks, num_features), np.nan)

    for i, ticker in enumerate(tickers):
        for j, feat in enumerate(feature_cols):
            col = f"{ticker}__{feat}"
            if col in df.columns:
                data[:, i, j] = df[col].values

    max_pred = max(pred_steps)
    total_samples = len(df) - sequence_length - max_pred
    X, Y, mask = [], [], []
    date_indices = []

    dates = df.index.to_list()

    for t in tqdm(range(total_samples), desc="Creating sequences"):
        seq = data[t : t + sequence_length]
        seq = np.nan_to_num(seq, nan=0.0)

        targets = []
        masks = []
        for step in pred_steps:
            target = data[t + sequence_length + step - 1, :, feature_cols.index(target_feature)]
            targets.append(np.nan_to_num(target, nan=0.0))
            masks.append(~np.isnan(target))

        Y.append(np.stack(targets, axis=-1))
        mask.append(np.all(masks, axis=0))
        X.append(seq)
        date_indices.append(dates[t + sequence_length - 1])

    X = np.array(X)
    Y = np.array(Y)
    mask = np.array(mask)
    date_indices = np.array(date_indices)

    X = X.reshape(X.shape[0], X.shape[1], -1)

    return X, Y, mask, date_indices


def train_test_split(
    X: np.ndarray, Y: np.ndarray, mask: np.ndarray, date_indices: np.ndarray,
    test_ratio: float = 0.2, drop_last_weeks: int = 52
) -> Tuple[dict, dict, dict]:
    num_samples = X.shape[0]
    split_point = int(num_samples * (1 - test_ratio))

    if drop_last_weeks >= (num_samples - split_point):
        raise ValueError("drop_last_weeks is too large, resulting in an empty test set.")

    train = {
        "X": X[:split_point],
        "Y": Y[:split_point],
        "mask": mask[:split_point],
        "dates": date_indices[:split_point],
    }

    test = {
        "X": X[split_point:-drop_last_weeks],
        "Y": Y[split_point:-drop_last_weeks],
        "mask": mask[split_point:-drop_last_weeks],
        "dates": date_indices[split_point:-drop_last_weeks],
    }

    print(f"Train: {train['X'].shape[0]} samples | Test: {test['X'].shape[0]} samples")
    return train, test, {"split_point": split_point, "drop_last_weeks": drop_last_weeks}


def save_data(train: dict, test: dict, tickers: List[str], features: List[str], output_path: str):
    os.makedirs(output_path, exist_ok=True)

    for name, data in zip(
        ['train_X', 'train_Y', 'train_mask', 'test_X', 'test_Y', 'test_mask'],
        [train['X'], train['Y'], train['mask'], test['X'], test['Y'], test['mask']]
    ):
        np.save(os.path.join(output_path, f"{name}.npy"), data)

    np.save(os.path.join(output_path, "tickers.npy"), np.array(tickers))
    np.save(os.path.join(output_path, "features.npy"), np.array(features))

    # Save train-test date info
    train_start = train["dates"][0]
    train_end = train["dates"][-1]
    test_start = test["dates"][0]
    test_end = test["dates"][-1]

    with open(os.path.join(output_path, "train_test_dates.txt"), "w") as f:
        f.write(f"Train Start: {train_start}\n")
        f.write(f"Train End:   {train_end}\n")
        f.write(f"Test Start:  {test_start}\n")
        f.write(f"Test End:    {test_end}\n")

    print("Saved all data and train/test dates to:", output_path)


def prepare_data(data_folder: str, output_folder: str):
    df_pivot, tickers, feature_cols = load_data(data_folder)
    X, Y, mask, date_indices = make_sequences_targets(df_pivot, tickers, feature_cols)
    train, test, _ = train_test_split(X, Y, mask, date_indices)
    save_data(train, test, tickers, feature_cols, output_folder)


if __name__ == "__main__":
    prepare_data(
        data_folder=r"data\processed_nifty50_data",
        output_folder=r"data\patchTST_data"
    )