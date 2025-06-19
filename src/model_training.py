import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import os

# --- CONFIG ---
CONFIG = {
    "DATA_DIR": "data/patchTST_data",
    "MODEL_PATH": "model/patchtst_model.pt",
    "BATCH_SIZE": 64,
    "EPOCHS": 20,
    "LEARNING_RATE": 1e-4,
    "WEIGHT_DECAY": 1e-4,
    "PATCH_SIZE": 16,
    "D_MODEL": 64,
    "N_HEADS": 8,
    "NUM_LAYERS": 4,
    "DROPOUT": 0.1,
    "CRITERION": "huber",  # or "directional"
    "EARLY_STOPPING_PATIENCE": 3,
    "VAL_SPLIT": 0.1,
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DirectionalMSELoss(nn.Module):
    def __init__(self, direction_penalty: float = 3.0):
        super().__init__()
        self.direction_penalty = direction_penalty

    def forward(self, preds, targets):
        mse = (preds - targets) ** 2
        wrong_sign = torch.sign(preds) != torch.sign(targets)
        penalty = torch.where(wrong_sign, torch.tensor(self.direction_penalty, device=preds.device), 1.0)
        return (mse * penalty).mean()


class PatchTST(nn.Module):
    def __init__(self, input_length, pred_len, num_features,
                 patch_size, d_model, n_heads, num_layers, dropout):
        super().__init__()
        self.patch_size = patch_size
        self.pred_len = pred_len
        self.num_features = num_features
        self.input_length = input_length
        self.num_patches = input_length // patch_size
        self.d_model = d_model

        self.patch_embed = nn.Sequential(
            nn.Linear(patch_size * num_features, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.time_pos_encoding = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        self.layernorm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, pred_len)
        )

    def forward(self, x):
        B, L, S, F = x.shape
        x = x.permute(0, 2, 1, 3)  # (B, S, L, F)
        x = x.unfold(2, self.patch_size, self.patch_size)  # (B, S, num_patches, patch_size, F)
        x = x.contiguous().view(B * S, self.num_patches, self.patch_size * F)  # (B*S, num_patches, patch_size*F)

        x = self.patch_embed(x) + self.time_pos_encoding  # (B*S, num_patches, d_model)
        x = self.layernorm(x)
        x = self.transformer(x)  # (B*S, num_patches, d_model)
        x = x.mean(dim=1)  # (B*S, d_model)
        out = self.head(x)  # (B*S, pred_len)
        out = out.view(B, S, self.pred_len)  # (B, S, pred_len)
        return out


def load_data():
    train_X = torch.tensor(np.load(f"{CONFIG['DATA_DIR']}/train_X.npy"), dtype=torch.float32)
    train_Y = torch.tensor(np.load(f"{CONFIG['DATA_DIR']}/train_Y.npy"), dtype=torch.float32)
    tickers = np.load(f"{CONFIG['DATA_DIR']}/tickers.npy")
    features = np.load(f"{CONFIG['DATA_DIR']}/features.npy")

    num_stocks = len(tickers)
    num_features = len(features)
    train_X = train_X.view(-1, train_X.shape[1], num_stocks, num_features)

    return train_X, train_Y, tickers, num_features


def train_model():
    train_X, train_Y, tickers, num_features = load_data()
    total_samples = train_X.shape[0]
    val_size = int(CONFIG["VAL_SPLIT"] * total_samples)

    # === Chronological Split ===
    train_X_part, val_X_part = train_X[:-val_size], train_X[-val_size:]
    train_Y_part, val_Y_part = train_Y[:-val_size], train_Y[-val_size:]

    train_ds = TensorDataset(train_X_part, train_Y_part)
    val_ds = TensorDataset(val_X_part, val_Y_part)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["BATCH_SIZE"])

    L = train_X.shape[1]
    T = train_Y.shape[2]
    model = PatchTST(
        input_length=L,
        pred_len=T,
        num_features=num_features,
        patch_size=CONFIG["PATCH_SIZE"],
        d_model=CONFIG["D_MODEL"],
        n_heads=CONFIG["N_HEADS"],
        num_layers=CONFIG["NUM_LAYERS"],
        dropout=CONFIG["DROPOUT"]
    ).to(DEVICE)

    if CONFIG["CRITERION"] == "directional":
        criterion = DirectionalMSELoss()
    else:
        criterion = nn.HuberLoss()

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LEARNING_RATE"], weight_decay=CONFIG["WEIGHT_DECAY"])

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(CONFIG["EPOCHS"]):
        model.train()
        total_loss = 0
        loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']}")
        for X, Y in loader:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loader.set_postfix(loss=loss.item())
        avg_train_loss = total_loss / len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, Y in val_loader:
                X, Y = X.to(DEVICE), Y.to(DEVICE)
                preds = model(X)
                loss = criterion(preds, Y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # Early Stopping Check
        if avg_val_loss < best_val_loss - 1e-6 :
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), CONFIG["MODEL_PATH"])
            print(" Saved new best model")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["EARLY_STOPPING_PATIENCE"]:
                print(" Early stopping triggered")
                break

    print(f" Training complete. Best Val Loss = {best_val_loss:.4f}")


if __name__ == "__main__":
    train_model()