import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm
import warnings
import json

warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self, seq_len: int, horizons: List[int]):
        self.seq_len = seq_len
        self.horizons = horizons
        self.max_horizon = max(horizons)

    def load_data(self, data_folder: str) -> Tuple[pd.DataFrame, List[str], List[str], pd.Timestamp, pd.Timestamp]:
        csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
        if not csv_files:
            raise ValueError(f"No CSV files found in {data_folder}")

        all_data, tickers = [], []

        for file in tqdm(csv_files, desc="Loading CSVs"):
            ticker = file.replace('.csv', '')
            tickers.append(ticker)
            file_path = os.path.join(data_folder, file)
            df = pd.read_csv(file_path, parse_dates=['Date'])
            if 'Date' not in df.columns:
                raise ValueError(f"'Date' column missing in {file}")
            df['Ticker'] = ticker
            df = df.sort_values('Date').reset_index(drop=True)
            all_data.append(df)

        df_all = pd.concat(all_data, ignore_index=True).sort_values(['Date', 'Ticker']).reset_index(drop=True)
        feature_cols = [col for col in df_all.columns if col not in ['Date', 'Ticker']]
        df_pivot = df_all.pivot(index='Date', columns='Ticker', values=feature_cols).sort_index()
        df_pivot.columns = [f"{ticker}__{feat}" for feat, ticker in df_pivot.columns]
        return df_pivot, sorted(tickers), feature_cols, df_pivot.index.min(), df_pivot.index.max()

    def create_sequences_and_targets(self, df: pd.DataFrame, tickers: List[str], feature_cols: List[str], target_feature: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if target_feature not in feature_cols:
            raise ValueError(f"Target feature '{target_feature}' not found in features: {feature_cols}")

        num_stocks, num_features = len(tickers), len(feature_cols)
        data = np.full((len(df), num_stocks, num_features), np.nan)

        for i, ticker in enumerate(tickers):
            for j, feat in enumerate(feature_cols):
                col = f"{ticker}__{feat}"
                if col in df.columns:
                    data[:, i, j] = df[col].values

        total_samples = len(df) - self.seq_len + 1
        if total_samples <= 0:
            raise ValueError(f"Not enough data. Need at least {self.seq_len} samples, got {len(df)}")

        X, Y, mask, date_indices = [], [], [], []
        dates = df.index.to_list()
        target_idx = feature_cols.index(target_feature)

        for t in tqdm(range(total_samples), desc="Creating sequences"):
            seq_start, seq_end = t, t + self.seq_len
            seq = np.nan_to_num(data[seq_start:seq_end], nan=0.0)
            targets, masks = [], []

            for horizon in self.horizons:
                # Calculate cumulative return over the horizon period
                cumulative_target, mask_h = self._calculate_cumulative_return(
                    data, seq_end, horizon, target_idx, num_stocks
                )
                targets.append(cumulative_target)
                masks.append(mask_h)

            Y.append(np.stack(targets, axis=-1))
            mask.append(np.all(masks, axis=0))
            X.append(seq)
            date_indices.append(dates[seq_end - 1])

        return np.array(X), np.array(Y), np.array(mask), np.array(date_indices)

    def _calculate_cumulative_return(self, data: np.ndarray, seq_end: int, horizon: int, target_idx: int, num_stocks: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate cumulative log return over the horizon period.
        For log returns, cumulative return = sum of log returns over the period.
        """
        target_start = seq_end
        target_end = seq_end + horizon
        
        if target_end > len(data):
            # Not enough future data available
            cumulative_target = np.zeros(num_stocks)
            mask_h = np.zeros(num_stocks, dtype=bool)
            return cumulative_target, mask_h
        
        # Extract the log returns for the horizon period
        period_returns = data[target_start:target_end, :, target_idx]  # Shape: (horizon, num_stocks)
        
        # Check for valid data (non-NaN values)
        valid_mask = ~np.isnan(period_returns)
        
        # For each stock, calculate cumulative log return only if all returns in the period are valid
        cumulative_target = np.zeros(num_stocks)
        mask_h = np.zeros(num_stocks, dtype=bool)
        
        for stock_idx in range(num_stocks):
            stock_returns = period_returns[:, stock_idx]
            stock_valid_mask = valid_mask[:, stock_idx]
            
            # Only calculate cumulative return if all returns in the period are valid
            if np.all(stock_valid_mask):
                # For log returns: cumulative return = sum of log returns
                cumulative_target[stock_idx] = np.sum(stock_returns)
                mask_h[stock_idx] = True
            else:
                cumulative_target[stock_idx] = 0.0
                mask_h[stock_idx] = False
        
        return cumulative_target, mask_h

    def temporal_train_val_test_split(self, X, Y, mask, date_indices, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15) -> Dict:
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, val, and test ratios must sum to 1.0")

        X_full, Y_full, mask_full, dates_full = X, Y, mask, date_indices
        usable_samples = X.shape[0] - (self.max_horizon - 1)

        X_valid, Y_valid, mask_valid, dates_valid = X[:usable_samples], Y[:usable_samples], mask[:usable_samples], date_indices[:usable_samples]
        num_valid = usable_samples
        train_end, val_end = int(num_valid * train_ratio), int(num_valid * (train_ratio + val_ratio))

        return {
            'train': {'X': X_valid[:train_end], 'Y': Y_valid[:train_end], 'mask': mask_valid[:train_end], 'dates': dates_valid[:train_end]},
            'val': {'X': X_valid[train_end:val_end], 'Y': Y_valid[train_end:val_end], 'mask': mask_valid[train_end:val_end], 'dates': dates_valid[train_end:val_end]},
            'test': {'X': X_valid[val_end:], 'Y': Y_valid[val_end:], 'mask': mask_valid[val_end:], 'dates': dates_valid[val_end:]},
            'full': {'X': X_full, 'Y': Y_full, 'mask': mask_full, 'dates': dates_full},
            'metadata': {
                'train_end_idx': train_end,
                'val_end_idx': val_end,
                'total_samples': num_valid,
                'train_end_date': dates_valid[train_end - 1],
                'val_end_date': dates_valid[val_end - 1],
                'test_end_date': dates_valid[-1]
            }
        }

    def save_processed_data(self, splits: Dict, tickers: List[str], features: List[str], output_path: str, target_feature: str):
        os.makedirs(output_path, exist_ok=True)
        for split_name in ['train', 'val', 'test', 'full']:
            for data_type in ['X', 'Y', 'mask', 'dates']:
                if data_type in splits[split_name]:
                    np.save(os.path.join(output_path, f"{split_name}_{data_type}.npy"), splits[split_name][data_type])

        np.save(os.path.join(output_path, "tickers.npy"), np.array(tickers))
        np.save(os.path.join(output_path, "features.npy"), np.array(features))

        metadata = splits['metadata']
        with open(os.path.join(output_path, "split_info.txt"), "w") as f:
            for k, v in metadata.items():
                f.write(f"{k}: {v}\n")

        config = {
            'seq_len': self.seq_len,
            'horizons': self.horizons,
            'max_horizon': self.max_horizon,
            'num_tickers': len(tickers),
            'num_features': len(features),
            'feature_names': features,
            'ticker_names': tickers,
            'data_start_date': metadata['train_end_date'].strftime('%Y-%m-%d'),
            'data_end_date': metadata['test_end_date'].strftime('%Y-%m-%d'),
            'target_feature': target_feature,
            'target_type': 'cumulative_log_return'
        }

        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        print(f"Saved all data to {output_path}")
        print(f"Target calculation: Cumulative log returns over horizons {self.horizons}")

def prepare_data(data_folder: str, output_folder: str, seq_len: int, horizons: List[int], target_feature: str):
    processor = DataProcessor(seq_len=seq_len, horizons=horizons)
    
    df_pivot, tickers, feature_cols, data_start_date, data_end_date = processor.load_data(data_folder)
    X, Y, mask, date_indices = processor.create_sequences_and_targets(df_pivot, tickers, feature_cols, target_feature)
    splits = processor.temporal_train_val_test_split(X, Y, mask, date_indices)
    splits['metadata']['data_start_date'] = data_start_date
    splits['metadata']['data_end_date'] = data_end_date
    processor.save_processed_data(splits, tickers, feature_cols, output_folder, target_feature)
    return splits, tickers, feature_cols

if __name__ == "__main__":
    prepare_data(
        data_folder=r"data/processed_nifty50_data",
        output_folder=r"data/model_input",
        seq_len=120,
        horizons=[20, 60],
        target_feature="log_return_Close"
    )