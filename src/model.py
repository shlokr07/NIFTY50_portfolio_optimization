import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os

def masked_mse_loss(pred, target, mask):
    mask = mask.unsqueeze(-1)  # [B, S, 1]
    loss = (pred - target) ** 2 * mask
    denom = mask.sum()
    if denom == 0:
        return torch.tensor(0.0, requires_grad=True, device=pred.device)
    return loss.sum() / denom

class FeatureEncoder(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(in_features, hidden_dim)

    def forward(self, x):
        # x: [B, T, S, F]
        return self.linear(x)  # [B, T, S, D]

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len, embed_dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, 1, embed_dim))

    def forward(self, x):
        return x + self.pos_embedding  # [B, T, S, D]

class TransformerModel(nn.Module):
    def __init__(self, num_stocks, num_features, seq_len, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.encoder = FeatureEncoder(num_features, d_model)
        self.pos_encoding = LearnablePositionalEncoding(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # For inter-stock attention at final time step
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # Final prediction head (per stock)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2)  # [1M, 3M]
        )

    def forward(self, x, mask=None):
        # x: [B, T, S, F]
        B, T, S, F = x.shape
        x = self.encoder(x)              # [B, T, S, D]
        x = self.pos_encoding(x)         # [B, T, S, D]

        # Reshape for temporal encoder
        x = x.permute(0, 2, 1, 3)        # [B, S, T, D]
        x = x.reshape(B * S, T, -1)      # [B*S, T, D]
        x = self.temporal_encoder(x)     # [B*S, T, D]
        x = x[:, -1, :]                  # Take final time step: [B*S, D]
        x = x.view(B, S, -1)             # [B, S, D]

        # Cross-stock attention (optional)
        # Allow stocks to see each other
        x, _ = self.cross_attn(x, x, x, key_padding_mask=~mask if mask is not None else None)  # [B, S, D]

        # Predict 1M and 3M returns per stock
        y = self.fc_out(x)               # [B, S, 2]
        return y

# --- Load Data ---
def load_tensor_data(path):
    def to_tensor(name):
        return torch.tensor(np.load(f"{path}/{name}.npy"), dtype=torch.float32)
    X = to_tensor("train_X")
    Y = to_tensor("train_Y")
    mask = to_tensor("train_mask").bool()
    return X, Y, mask

train_X, train_Y, train_mask = load_tensor_data("data/train_test")
test_X, test_Y, test_mask = load_tensor_data("data/train_test")
print("Any NaNs in train_X?", torch.isnan(train_X).any().item())
print("Any NaNs in train_Y?", torch.isnan(train_Y).any().item())

# --- Create Dataloaders ---
BATCH_SIZE = 16
train_ds = TensorDataset(train_X, train_Y, train_mask)
test_ds = TensorDataset(test_X, test_Y, test_mask)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

# --- Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(num_stocks=47, num_features=train_X.shape[-1], seq_len=train_X.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Reshape mask to match train_X
mask_expanded = train_mask.unsqueeze(1).unsqueeze(-1)  # [858, 1, 47, 1]
mask_expanded = mask_expanded.expand(-1, train_X.shape[1], -1, train_X.shape[3])  # [858, 12, 47, 16]

# Check for NaNs only where mask = True
has_nan = torch.isnan(train_X) & mask_expanded
if has_nan.any():
    print("❌ NaNs detected in valid (masked=True) training inputs!")
else:
    print("✅ No NaNs in valid training inputs.")

# --- Training Loop ---
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for Xb, Yb, Mb in train_dl:
        Xb, Yb, Mb = Xb.to(device), Yb.to(device), Mb.to(device)
        optimizer.zero_grad()
        preds = model(Xb, Mb)
        loss = masked_mse_loss(preds, Yb, Mb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * Xb.size(0)

    train_loss /= len(train_dl.dataset)
    print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.6f}")

# --- Final Evaluation ---
model.eval()
all_preds = []
all_targets = []
all_masks = []

with torch.no_grad():
    for Xb, Yb, Mb in test_dl:
        Xb, Yb, Mb = Xb.to(device), Yb.to(device), Mb.to(device)
        preds = model(Xb, Mb)
        all_preds.append(preds.cpu())
        all_targets.append(Yb.cpu())
        all_masks.append(Mb.cpu())

all_preds = torch.cat(all_preds, dim=0)     # [N, S, 2]
all_targets = torch.cat(all_targets, dim=0) # [N, S, 2]
all_masks = torch.cat(all_masks, dim=0)     # [N, S]

# --- Load tickers ---
tickers = np.load("data/train_test/tickers.npy")

# --- Print final predictions ---
print("\n--- Final Evaluation on Test Set ---")
for i in range(all_preds.shape[0]):
    print(f"\nSample {i + 1}")
    for j in range(all_preds.shape[1]):
        if all_masks[i, j]:
            pred_1m, pred_3m = all_preds[i, j].tolist()
            target_1m, target_3m = all_targets[i, j].tolist()
            ticker = tickers[j] if j < len(tickers) else f"Stock{j}"
            print(f"  {ticker:15s} | Pred: [1M={pred_1m:.4f}, 3M={pred_3m:.4f}] | Target: [1M={target_1m:.4f}, 3M={target_3m:.4f}]")