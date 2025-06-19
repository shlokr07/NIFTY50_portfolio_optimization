# 📈 NIFTY50 Portfolio Optimization with PatchTST

This project uses deep learning to build a **momentum-based portfolio strategy** on NIFTY 50 constituents. It applies the **PatchTST (Transformer-based time series model)** to forecast **1-month and 3-month ahead returns** for each stock. A simple **top-5 rebalanced portfolio** is constructed from the predictions and compared against the NIFTY 50 benchmark.

---

## 🔢 Data Used

- **Universe**: 47 stocks that are *currently part of the NIFTY 50* index.
- **Features**:
  - Raw price and volume: `Open`, `High`, `Low`, `Close`, `Volume`
  - Derived features: `log_return_Close` and several Technical Indicators.

Sequences of 120 trading days (≈ 6 months) are used as input, and the model forecasts the **20-day and 60-day forward log returns** for each stock.

---

## 🔍 Project Overview

| Component           | Details                                                                 |
|---------------------|-------------------------------------------------------------------------|
| Model               | `PatchTST`: Transformer with patch-based time series embedding          |
| Targets             | 1-month (20-day) and 3-month (60-day) forward log returns                |
| Input Sequence      | 120 trading days (~6 months) per sample                                 |
| Loss Function       | Huber Loss or Directional MSE Loss (penalizes incorrect direction)       |
| Portfolio Strategy  | Long-only, daily top-5 rebalancing based on weighted prediction scores   |
| Benchmark           | Cumulative returns of the NIFTY 50 Index            |

---
