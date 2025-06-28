# model_eval.py

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_absolute_error
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from scipy.stats import pearsonr
from model_training import QuantileTransformer

warnings.filterwarnings("ignore")

CONFIG = {
    "DATA_DIR": "data/model_input",
    "MODEL_PATH": "model/_quantile_model.pt",
    "RESULTS_DIR": "results/evaluation",
    "FIGURES_DIR": "results/evaluation/figures"
}

def quantile_loss(pred, target, quantile):
    error = target - pred
    return torch.max((quantile - 1) * error, quantile * error).mean().item()

def compute_directional_accuracy(pred, true):
    return (np.sign(pred) == np.sign(true)).mean()

def compute_ic(pred, true):
    return pd.Series(pred).rank().corr(pd.Series(true).rank())

def compute_coverage(pred, true, q):
    return (true <= pred).mean()

def compute_residuals(pred, true):
    return true - pred

def evaluate_predictions(preds, Y, mask, quantiles, horizons):
    results = []
    for s in range(Y.shape[1]):
        valid_mask = mask[:, s]
        if valid_mask.sum() == 0:
            continue
        true_vals = Y[valid_mask, s, :]
        pred_vals = preds[valid_mask, s, :, :]
        for q_idx, q in enumerate(quantiles):
            for h_idx, h in enumerate(horizons):
                pred = pred_vals[:, q_idx, h_idx]
                true = true_vals[:, h_idx]
                res = {
                    "ticker": s,
                    "quantile": q,
                    "horizon": h,
                    "quantile_loss": quantile_loss(torch.tensor(pred), torch.tensor(true), q),
                    "coverage": compute_coverage(pred, true, q),
                    "mae": mean_absolute_error(true, pred),
                    "pearson": pearsonr(true, pred)[0] if len(true) > 1 else np.nan,
                    "directional_accuracy": compute_directional_accuracy(pred, true),
                    "ic": compute_ic(pred, true),
                }
                results.append(res)
    return pd.DataFrame(results)

def residual_diagnostics(preds, Y, mask, quantiles, horizons):
    diagnostics = []
    for s in range(Y.shape[1]):
        valid_mask = mask[:, s]
        if valid_mask.sum() == 0:
            continue
        true_vals = Y[valid_mask, s, :]
        pred_vals = preds[valid_mask, s, :, :]
        for q_idx, q in enumerate(quantiles):
            for h_idx, h in enumerate(horizons):
                pred = pred_vals[:, q_idx, h_idx]
                true = true_vals[:, h_idx]
                residuals = compute_residuals(pred, true)
                ljung_p = acorr_ljungbox(residuals, lags=[10], return_df=True)['lb_pvalue'].iloc[0]
                try:
                    het_p = het_breuschpagan(residuals[:, None], np.ones((len(residuals), 1)))[1]
                except:
                    het_p = np.nan
                diagnostics.append({
                    "ticker": s,
                    "quantile": q,
                    "horizon": h,
                    "ljung_box_pvalue": ljung_p,
                    "heteroscedasticity_pvalue": het_p
                })
    return pd.DataFrame(diagnostics)

def plot_actual_vs_predicted(preds, Y, mask, quantiles, horizons, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for q_idx, q in enumerate(quantiles):
        for h_idx, h in enumerate(horizons):
            all_true, all_pred = [], []
            for s in range(Y.shape[1]):
                valid_mask = mask[:, s]
                if valid_mask.sum() == 0:
                    continue
                true_vals = Y[valid_mask, s, h_idx]
                pred_vals = preds[valid_mask, s, q_idx, h_idx]
                all_true.extend(true_vals)
                all_pred.extend(pred_vals)
            plt.figure(figsize=(6, 6))
            plt.scatter(all_true, all_pred, alpha=0.3)
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title(f"Actual vs Predicted (Q={q}, H={h})")
            plt.plot([min(all_true), max(all_true)], [min(all_true), max(all_true)], 'r--')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"scatter_q{q}_h{h}.png"))
            plt.close()

def plot_temporal_predictions(preds, Y, mask, quantiles, horizons, output_dir="results/evaluation/figures", dates=None):
    os.makedirs(output_dir, exist_ok=True)
    B, S, Q, H = preds.shape

    for s in range(S):  # For each stock
        for h_idx, h in enumerate(horizons):
            valid_mask = mask[:, s].astype(bool)
            if valid_mask.sum() == 0:
                continue

            true_series = Y[valid_mask, s, h_idx]
            date_series = np.array(dates)[valid_mask] if dates is not None else np.arange(len(true_series))

            plt.figure(figsize=(14, 5))
            plt.plot(date_series, true_series, label="True", color="black", linewidth=1.2)

            for q_idx, q in enumerate(quantiles):
                pred_series = preds[valid_mask, s, q_idx, h_idx]
                label = f"Q={q}"
                color = {0.1: "red", 0.5: "orange", 0.9: "green"}.get(q, f"C{q_idx}")
                style = "--" if q in [0.1, 0.9] else "-"
                plt.plot(date_series, pred_series, label=label, color=color, linestyle=style, alpha=0.8)

            plt.xlabel("Date")
            plt.ylabel("Return")
            plt.title(f"Temporal Quantile Predictions (Stock={s}, Horizon={h})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            filename = os.path.join(output_dir, f"time_series_allQ_ticker{s}_h{h}.png")
            plt.savefig(filename, dpi=150)
            plt.close()

def debug_stock_predictions(preds, Y, mask, tickers, stage="test"):
    B, S, Q, H = preds.shape
    print(f"\n=== DEBUG: {stage.upper()} SET ===")
    print(f"Shape: B={B}, S={S}, Q={Q}, H={H}")

    valid_counts = mask.sum(dim=0).cpu().numpy()
    q50 = preds[:, :, 1, :]
    std_time = q50.std(dim=0).mean(dim=-1).cpu().numpy()
    spread = (preds[:, :, 2, :] - preds[:, :, 0, :]).abs().mean(dim=0).mean(dim=-1).cpu().numpy()

    print("\nStock Valid Counts | Q=0.5 STD | Q0.9-Q0.1 Spread")
    for i in range(S):
        print(f"{tickers[i]:>8}: {valid_counts[i]:>5} | {std_time[i]:.4f} | {spread[i]:.4f}")

    low_std_stocks = [tickers[i] for i in range(S) if std_time[i] < 0.002]
    low_spread_stocks = [tickers[i] for i in range(S) if spread[i] < 0.002]

    print(f"\n❗ Flat Predictions (std < 0.002): {low_std_stocks}")
    print(f"Low Quantile Spread (spread < 0.002): {low_spread_stocks}")

def main():
    os.makedirs(CONFIG['RESULTS_DIR'], exist_ok=True)
    os.makedirs(CONFIG['FIGURES_DIR'], exist_ok=True)

    # Load model + architecture
    checkpoint = torch.load(CONFIG['MODEL_PATH'], map_location=torch.device("cpu"))
    arch = {k: v for k, v in checkpoint['model_architecture'].items() if k != 'class_name'}
    model = QuantileTransformer(**arch)
    state_dict = {
        (k.replace("encoder.layers", "encoder") if k.startswith("encoder.layers") else k): v
        for k, v in checkpoint['model_state_dict'].items()
    }
    model.load_state_dict(state_dict)
    model.eval()

    # Load data
    X = np.load(os.path.join(CONFIG['DATA_DIR'], "test_X.npy"))
    Y = np.load(os.path.join(CONFIG['DATA_DIR'], "test_Y.npy"))
    mask = np.load(os.path.join(CONFIG['DATA_DIR'], "test_mask.npy"))
    tickers = np.load(os.path.join(CONFIG['DATA_DIR'], "tickers.npy"))
    try:
        test_dates = np.load(os.path.join(CONFIG['DATA_DIR'], "test_dates.npy"), allow_pickle=True)
    except FileNotFoundError:
        test_dates = None

    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32)).cpu()

    quantiles = checkpoint['model_architecture']['quantiles']
    horizons = list(range(preds.shape[-1]))

    # Evaluate
    metrics_df = evaluate_predictions(preds.numpy(), Y, mask, quantiles, horizons)
    diagnostics_df = residual_diagnostics(preds.numpy(), Y, mask, quantiles, horizons)
    plot_actual_vs_predicted(preds.numpy(), Y, mask, quantiles, horizons, CONFIG['FIGURES_DIR'])
    plot_temporal_predictions(preds.numpy(), Y, mask, quantiles, horizons, CONFIG['FIGURES_DIR'], test_dates)

    # Save
    metrics_df.to_csv(os.path.join(CONFIG['RESULTS_DIR'], "quantile_metrics.csv"), index=False)
    diagnostics_df.to_csv(os.path.join(CONFIG['RESULTS_DIR'], "residual_diagnostics.csv"), index=False)

    # Debug analysis
    debug_stock_predictions(preds, Y=torch.tensor(Y), mask=torch.tensor(mask), tickers=tickers, stage="test")

    print("Evaluation complete. Results in:", CONFIG['RESULTS_DIR'])

if __name__ == "__main__":
    main()
