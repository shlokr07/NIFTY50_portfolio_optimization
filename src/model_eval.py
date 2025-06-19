import os
import json
import torch
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

from model_training import PatchTST, CONFIG

# --- Config ---
DATA_DIR = "data/patchTST_data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Wm1 = 0.3  # Weight for 1-month prediction
Wm3 = 0.7  # Weight for 3-month prediction

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

def evaluate(model, test_X, test_Y, tickers):
    B, L, S, F = test_X.shape
    T = test_Y.shape[2]
    test_ds = TensorDataset(test_X, test_Y)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["BATCH_SIZE"])

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, Y in tqdm(test_loader, desc="Evaluating"):
            X = X.to(DEVICE)
            preds = model(X)
            all_preds.append(preds.cpu())
            all_targets.append(Y)

    preds = torch.cat(all_preds).numpy()  # [B, S, T]
    targets = torch.cat(all_targets).numpy()  # [B, S, T]

    results = []
    for s in range(S):
        result = {"Ticker": tickers[s]}
        for t in range(T):
            pred = preds[:, s, t]
            true = targets[:, s, t]

            mse = mean_squared_error(true, pred)
            mae = mean_absolute_error(true, pred)
            corr, _ = pearsonr(true, pred)
            da = np.mean(np.sign(true) == np.sign(pred))
            sim = cosine_similarity(true.reshape(1, -1), pred.reshape(1, -1))[0, 0]

            result[f"MSE_T{t+1}"] = mse
            result[f"MAE_T{t+1}"] = mae
            result[f"Corr_T{t+1}"] = corr
            result[f"DA_T{t+1}"] = da
            result[f"Sim_T{t+1}"] = sim
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv("evaluation_metrics.csv", index=False)
    print("Saved evaluation metrics to evaluation_metrics.csv")

    # Averages
    avg_corr = df[[col for col in df.columns if "Corr" in col]].mean().mean()
    avg_da = df[[col for col in df.columns if "DA" in col]].mean().mean()
    avg_sim = df[[col for col in df.columns if "Sim" in col]].mean().mean()

    print("\nAverage Model Performance Across All Stocks:")
    print(f"Average Pearson Correlation: {avg_corr:.4f}")
    print(f"Average Directional Accuracy: {avg_da:.4f}")
    print(f"Average Cosine Similarity: {avg_sim:.4f}")

def backtest(model, test_X, test_Y, tickers):
    input_len = test_X.shape[1]
    pred_len = test_Y.shape[2]
    B, S = test_X.shape[0], test_X.shape[2]

    # Predict again
    test_ds = TensorDataset(test_X, test_Y)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["BATCH_SIZE"])

    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, Y in test_loader:
            X = X.to(DEVICE)
            preds = model(X)
            all_preds.append(preds.cpu())
            all_targets.append(Y)

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    # Weighted signal
    weights = np.array([Wm1, Wm3])
    weighted_preds = np.tensordot(preds, weights, axes=([2], [0]))
    weighted_targets = np.tensordot(targets, weights, axes=([2], [0]))

    # Backtest
    print("\nRunning Daily Rebalanced Backtest (Top 5 Portfolio)...")
    test_start_date, _ = load_test_dates()
    test_dates = pd.bdate_range(start=test_start_date, periods=B)

    topk = 5
    cumulative_log_return = 0.0
    strategy_returns = []
    portfolio_history = {}

    for i in range(B):
        top_indices = set(np.argsort(weighted_preds[i])[-topk:])
        daily_return = weighted_targets[i, list(top_indices)].mean()
        cumulative_log_return += daily_return
        strategy_returns.append(np.exp(cumulative_log_return))
        portfolio_history[str(test_dates[i].date())] = [tickers[idx] for idx in top_indices]

    with open("portfolio_history.json", "w") as f:
        json.dump(portfolio_history, f, indent=2)
    print("Saved portfolio history to portfolio_history.json")

    # Fetch NIFTY 50
    print("Fetching NIFTY 50...")
    nifty = yf.download("^NSEI", start=str(test_dates[0].date()), end=str(test_dates[-1].date()))
    nifty = np.log(nifty["Close"]).diff().fillna(0)
    nifty_cum = np.exp(np.cumsum(nifty.values))

    # Align
    strategy_returns = np.array(strategy_returns)
    min_len = min(len(strategy_returns), len(nifty_cum))
    aligned_dates = test_dates[:min_len]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(aligned_dates, strategy_returns[:min_len], label="PatchTST Strategy")
    plt.plot(aligned_dates, nifty_cum[:min_len], label="NIFTY 50", linestyle="--")
    plt.title("Cumulative Returns (Log): PatchTST Strategy vs NIFTY 50")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)

    os.makedirs("plots", exist_ok=True)
    plot_path = f"plots/strategy_vs_nifty_Wm1-{Wm1}_Wm3-{Wm3}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    test_X, test_Y, tickers, num_features = load_data()
    input_len = test_X.shape[1]
    pred_len = test_Y.shape[2]

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

    model.load_state_dict(torch.load(CONFIG["MODEL_PATH"], map_location=DEVICE))
    model.eval()
    print(f"Loaded model from {CONFIG['MODEL_PATH']}")

    evaluate(model, test_X, test_Y, tickers)
    backtest(model, test_X, test_Y, tickers)