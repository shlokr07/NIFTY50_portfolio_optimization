import yfinance as yf
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from tqdm import tqdm
import os
import json
from model_training import PatchTST, CONFIG

# --- Config ---
DATA_DIR = "data/patchTST_data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Prediction weights
Wm1 = 0.7 
Wm3 = 0.3

def load_data():
    test_X = torch.tensor(np.load(f"{DATA_DIR}/test_X.npy"), dtype=torch.float32)
    test_Y = torch.tensor(np.load(f"{DATA_DIR}/test_Y.npy"), dtype=torch.float32)
    tickers = np.load(f"{DATA_DIR}/tickers.npy")
    features = np.load(f"{DATA_DIR}/features.npy")

    num_stocks = len(tickers)
    num_features = len(features)
    test_X = test_X.view(-1, test_X.shape[1], num_stocks, num_features)

    return test_X, test_Y, tickers, num_features


def load_test_dates():
    date_file = os.path.join(DATA_DIR, "train_test_dates.txt")
    with open(date_file, "r") as f:
        lines = f.readlines()
    date_dict = {line.split(":")[0].strip(): pd.to_datetime(line.split(":")[1].strip()) for line in lines}
    return date_dict["Test Start"], date_dict["Test End"]


def evaluate_and_backtest():
    test_X, test_Y, tickers, num_features = load_data()
    test_ds = TensorDataset(test_X, test_Y)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['BATCH_SIZE'])

    input_len = test_X.shape[1]
    pred_len = test_Y.shape[2]
    B, S = test_X.shape[0], test_X.shape[2]

    model = PatchTST(
        input_length=input_len,
        pred_len=pred_len,
        num_features=num_features,
        patch_size=CONFIG["PATCH_SIZE"],
        d_model=CONFIG["D_MODEL"],
        n_heads=CONFIG["N_HEADS"],
        num_layers=CONFIG["NUM_LAYERS"],
        dropout=CONFIG["DROPOUT"]
    ).to(DEVICE)

    model.load_state_dict(torch.load(CONFIG['MODEL_PATH'], map_location=DEVICE))
    model.eval()
    print(f"✅ Loaded model from {CONFIG['MODEL_PATH']}")

    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, Y in tqdm(test_loader, desc="Evaluating"):
            X = X.to(DEVICE)
            preds = model(X)
            all_preds.append(preds.cpu())
            all_targets.append(Y)

    preds = torch.cat(all_preds).numpy()  # [B, S, T]
    targets = torch.cat(all_targets).numpy()  # [B, S, T]

    # === Weighted Prediction (1M , 3M) ===
    weights = np.array([Wm1, Wm3])
    weighted_preds = np.tensordot(preds, weights, axes=([2], [0]))  # [B, S]
    weighted_targets = np.tensordot(targets, weights, axes=([2], [0]))  # [B, S]

    # === Evaluation ===
    flat_pred = weighted_preds.flatten()
    flat_true = weighted_targets.flatten()

    mse = mean_squared_error(flat_true, flat_pred)
    mae = mean_absolute_error(flat_true, flat_pred)
    da = np.mean(np.sign(flat_pred) == np.sign(flat_true))
    corr, _ = pearsonr(flat_pred, flat_true)
    tp = np.sum((flat_pred > 0) & (flat_true > 0))
    tn = np.sum((flat_pred <= 0) & (flat_true <= 0))
    fp = np.sum((flat_pred > 0) & (flat_true <= 0))
    fn = np.sum((flat_pred <= 0) & (flat_true > 0))

    print("\n📊 Evaluation Metrics:")
    print(f"MSE: {mse:.6f}, MAE: {mae:.6f}, Pearson Corr: {corr:.4f}, DA: {da:.4f}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    # === Backtest ===
    print("\n📈 Running Daily Rebalanced Backtest (Top 5 Portfolio)...")
    test_start_date, _ = load_test_dates()
    test_dates = pd.bdate_range(start=test_start_date, periods=B)

    topk = 5
    cumulative_log_return = 0.0
    strategy_returns = []
    portfolio_history = {}
    current_portfolio = set()

    for i in range(B):
        top_indices = set(np.argsort(weighted_preds[i])[-topk:])
        current_portfolio = top_indices  # rebalance
        daily_return = weighted_targets[i, list(current_portfolio)].mean()
        cumulative_log_return += daily_return
        strategy_returns.append(np.exp(cumulative_log_return))  # convert to compounding return
        portfolio_history[str(test_dates[i].date())] = [tickers[idx] for idx in current_portfolio]

    with open("portfolio_history.json", "w") as f:
        json.dump(portfolio_history, f, indent=2)
    print("📁 Saved portfolio history to portfolio_history.json")

    # === Fetch NIFTY 50 ===
    print("Fetching NIFTY 50...")
    nifty = yf.download("^NSEI", start=str(test_dates[0].date()), end=str(test_dates[-1].date()))
    nifty = np.log(nifty["Close"]).diff().fillna(0)
    nifty_cum = np.exp(np.cumsum(nifty.values))

    # === Align lengths ===
    strategy_returns = np.array(strategy_returns)
    min_len = min(len(strategy_returns), len(nifty_cum))
    strategy_returns = strategy_returns[:min_len]
    nifty_cum = nifty_cum[:min_len]
    aligned_dates = test_dates[:min_len]

    # === Plot ===
    plt.figure(figsize=(12, 6))
    plt.plot(aligned_dates, strategy_returns, label="PatchTST Strategy")
    plt.plot(aligned_dates, nifty_cum, label="NIFTY 50", linestyle="--")
    plt.title("Cumulative Returns (Log): PatchTST Strategy vs NIFTY 50")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/strategy_vs_nifty.png")
    plt.close()
    print("📊 Saved plot to plots/strategy_vs_nifty.png")


if __name__ == "__main__":
    evaluate_and_backtest()