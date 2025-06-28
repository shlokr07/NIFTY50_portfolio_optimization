# model_training.py 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    "DATA_DIR": r"data/model_input",
    "MODEL_PATH": r"model/_quantile_model.pt",
    "RESULTS_DIR": r"results/training",
    
    # Training parameters
    "EPOCHS": 30,
    "EARLY_STOPPING_PATIENCE": 5,
    "BATCH_SIZE": 64,
    "LEARNING_RATE": 1e-3,
    "WEIGHT_DECAY": 0.0,
    
    # Model architecture
    "PATCH_SIZE": 20,
    "D_MODEL": 128,
    "N_HEADS": 16,
    "NUM_LAYERS": 6,
    "DROPOUT": 0.0,
    "QUANTILES": [0.1, 0.5, 0.9],
    "NUM_HORIZONS": 2,

    "DECORR_WEIGHT":0.0,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class ResidualTransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer with Residual Skip"""
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return x + self.layer(self.norm(x))

class QuantileTransformer(nn.Module):
    """
    Transformer-based Quantile Forecaster with Non-Overlapping Patches.
    """
    def __init__(self, seq_len, num_stocks, num_features, patch_size, d_model, n_heads, num_layers, dropout, quantiles, num_horizons=3):
        super().__init__()
        self.seq_len = seq_len
        self.num_stocks = num_stocks
        self.num_features = num_features
        self.patch_size = patch_size
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.num_horizons = num_horizons
        self.d_model = d_model

        # Compute max number of non-overlapping patches
        self.max_patches = seq_len // patch_size

        # Projection layer from patch to d_model
        self.patch_embed = nn.Sequential(
            nn.Linear(num_features * patch_size, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # Temporal positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, self.max_patches, d_model) * 0.02)

        # Transformer encoder
        self.encoder = nn.Sequential(*[
            ResidualTransformerEncoderLayer(d_model, n_heads, dropout)
            for _ in range(num_layers)
        ])

        # Global pooling (temporal)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Horizon-specific heads
        self.horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, self.num_quantiles)
            ) for _ in range(num_horizons)
        ])

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, L, S, F]
        Returns:
            predictions: [B, S, Q, H]
        """
        B, L, S, F = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B * S, L, F)  # [B*S, L, F]

        # Cut to largest multiple of patch_size
        usable_len = (L // self.patch_size) * self.patch_size
        x = x[:, :usable_len, :]  # [B*S, usable_len, F]

        # Non-overlapping patches
        patches = x.view(B * S, usable_len // self.patch_size, self.patch_size * F)  # [B*S, num_patches, patch_size * F]
        patches = self.patch_embed(patches)  # [B*S, num_patches, d_model]

        # Add temporal positional encoding
        patches = patches + self.pos_encoding[:, :patches.size(1), :]  # broadcast position

        # Transformer encoder
        encoded = self.encoder(patches)  # [B*S, num_patches, d_model]

        # Global average pooling across patches
        x_pooled = encoded.transpose(1, 2)  # [B*S, d_model, num_patches]
        x_summary = self.global_pool(x_pooled).squeeze(-1)  # [B*S, d_model]
        x_summary = x_summary.view(B, S, self.d_model)  # [B, S, d_model]

        # Quantile predictions from each head
        outputs = [head(x_summary) for head in self.horizon_heads]  # list of [B, S, Q]
        return torch.stack(outputs, dim=-1)  # [B, S, Q, H]

# -----------------------------
# Custom Quantile Loss with Residual Decorrelation Only
# -----------------------------

class CustomLoss(nn.Module):
    def __init__(self, quantiles: List[float], delta: float = 1.0,
                 residual_decorrelation_weight: float = 0.1):
        super().__init__()
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32)
        self.delta = delta
        self.residual_decorrelation_weight = residual_decorrelation_weight

    def forward(self, preds, targets, mask=None):
        device = preds.device
        quantiles = self.quantiles.to(device)
        median_idx = len(quantiles) // 2
        median_preds = preds[:, :, median_idx, :]  # [B, S, H]

        targets_expanded = targets.unsqueeze(2)  # [B, S, 1, H]
        quantiles_expanded = quantiles.view(1, 1, -1, 1)
        errors = targets_expanded - preds
        quantile_losses = torch.max((quantiles_expanded - 1) * errors, quantiles_expanded * errors)

        if mask is not None:
            mask_expanded = mask.unsqueeze(2).unsqueeze(3).float()
            quantile_losses = quantile_losses * mask_expanded
            base_loss = quantile_losses.sum() / (mask_expanded.sum() * preds.shape[2] * preds.shape[3])
        else:
            base_loss = quantile_losses.mean()

        # Residual decorrelation penalty only
        residuals = median_preds - targets  # [B, S, H]
        B, S, H = residuals.shape
        residuals = residuals.permute(0, 2, 1).reshape(-1, S)  # [B*H, S]
        cov = torch.cov(residuals.T)
        off_diag = cov - torch.diag(torch.diag(cov))
        decorrelation_penalty = (off_diag ** 2).mean()

        return base_loss + self.residual_decorrelation_weight * decorrelation_penalty

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model_state = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta :
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

def load_training_data():
    """Load training and validation data"""
    data_dir = CONFIG["DATA_DIR"]
    
    # Load arrays
    train_X = np.load(f"{data_dir}/train_X.npy")
    train_Y = np.load(f"{data_dir}/train_Y.npy") 
    train_mask = np.load(f"{data_dir}/train_mask.npy")
    
    val_X = np.load(f"{data_dir}/val_X.npy")
    val_Y = np.load(f"{data_dir}/val_Y.npy")
    val_mask = np.load(f"{data_dir}/val_mask.npy")
    
    # Load metadata
    tickers = np.load(f"{data_dir}/tickers.npy")
    features = np.load(f"{data_dir}/features.npy")
    
    with open(f"{data_dir}/config.json", 'r') as f:
        data_config = json.load(f)
    
    # Convert to tensors
    train_X = torch.tensor(train_X, dtype=torch.float32)
    train_Y = torch.tensor(train_Y, dtype=torch.float32)
    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    
    val_X = torch.tensor(val_X, dtype=torch.float32)
    val_Y = torch.tensor(val_Y, dtype=torch.float32)
    val_mask = torch.tensor(val_mask, dtype=torch.bool)
    
    print(f"Training data - X: {train_X.shape}, Y: {train_Y.shape}, mask: {train_mask.shape}")
    print(f"Validation data - X: {val_X.shape}, Y: {val_Y.shape}, mask: {val_mask.shape}")
    
    return train_X, train_Y, train_mask, val_X, val_Y, val_mask, tickers, features, data_config

def create_data_loaders(train_X, train_Y, train_mask, val_X, val_Y, val_mask, batch_size):
    """Create data loaders"""
    train_dataset = TensorDataset(train_X, train_Y, train_mask)
    val_dataset = TensorDataset(val_X, val_Y, val_mask)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )
    
    return train_loader, val_loader

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch_idx, (X, Y, mask) in enumerate(progress_bar):
        torch.cuda.empty_cache()
        X, Y, mask = X.to(device), Y.to(device), mask.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(X)  # [B, S, num_quantiles, num_horizons]
        loss = criterion(outputs, Y, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg': f'{total_loss/num_batches:.4f}'
        })
    
    return total_loss / num_batches

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for X, Y, mask in tqdm(val_loader, desc="Validating"):
            torch.cuda.empty_cache()
            X, Y, mask = X.to(device), Y.to(device), mask.to(device)
            
            outputs = model(X)
            loss = criterion(outputs, Y, mask)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def save_model(model, model_info, config, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'model_info': model_info,
        'model_architecture': {
            'class_name': 'QuantileTransformer',
            'seq_len': model.seq_len,
            'num_stocks': model.num_stocks,
            'num_features': model.num_features,
            'patch_size': model.patch_size,
            'd_model': model.d_model,
            'n_heads': model.encoder[0].layer.self_attn.num_heads,
            'num_layers': len(model.encoder),
            'dropout': config['DROPOUT'],
            'quantiles': model.quantiles,
            'num_horizons': model.num_horizons
        }
    }, model_path)

    print(f"Model saved to: {model_path}")

def train_model():
    """Main training function"""
    print(f"{'='*80}")
    print(f"TRAINING QUANTILE TRANSFORMER MODEL")
    
    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)
    
    train_X, train_Y, train_mask, val_X, val_Y, val_mask, tickers, features, data_config = load_training_data()
    train_loader, val_loader = create_data_loaders(
        train_X, train_Y, train_mask, val_X, val_Y, val_mask, CONFIG["BATCH_SIZE"]
    )

    seq_len, num_stocks, num_features = train_X.shape[1], train_X.shape[2], train_X.shape[3]
    
    model = QuantileTransformer(
        seq_len=seq_len,
        num_stocks=num_stocks,
        num_features=num_features,
        patch_size=CONFIG["PATCH_SIZE"],
        d_model=CONFIG["D_MODEL"],
        n_heads=CONFIG["N_HEADS"],
        num_layers=CONFIG["NUM_LAYERS"],
        dropout=CONFIG["DROPOUT"],
        quantiles=CONFIG["QUANTILES"],
        num_horizons=train_Y.shape[-1]
    ).to(DEVICE)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Initialize optimizer and loss
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=CONFIG["LEARNING_RATE"],
        weight_decay=CONFIG["WEIGHT_DECAY"]
    )
    
    # Updated criterion with only residual decorrelation
    criterion = CustomLoss(
        CONFIG["QUANTILES"],
        residual_decorrelation_weight=CONFIG['DECORR_WEIGHT']
    )
    
    early_stopping = EarlyStopping(CONFIG["EARLY_STOPPING_PATIENCE"])
    
    # Training loop
    train_losses = []
    val_losses = []
    
    print(f"\nStarting training for {CONFIG['EPOCHS']} epochs...")
    print(f"{'='*80}")
    
    for epoch in range(CONFIG["EPOCHS"]):
        print(f"\nEpoch {epoch+1}/{CONFIG['EPOCHS']}")
        print(f"-" * 40)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, DEVICE)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model
    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)
        print(f"\nLoaded best model with validation loss: {early_stopping.best_loss:.4f}")
    
    # Prepare model info for saving
    model_info = {
        'data_config': data_config,
        'tickers': tickers.tolist() if isinstance(tickers, np.ndarray) else tickers,
        'features': features.tolist() if isinstance(features, np.ndarray) else features,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': early_stopping.best_loss,
        'final_epoch': len(train_losses),
        'training_completed': True
    }
    
    # Save the trained model
    save_model(model, model_info, CONFIG, CONFIG["MODEL_PATH"])
    
    # Plot and save training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8, color='orange')
    plt.axhline(y=early_stopping.best_loss, color='red', linestyle='--', 
                label=f'Best Val Loss: {early_stopping.best_loss:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{CONFIG['RESULTS_DIR']}/training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save training summary
    summary = {
        'config': CONFIG,
        'model_info': model_info,
        'training_summary': {
            'total_epochs': len(train_losses),
            'best_val_loss': early_stopping.best_loss,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'model_parameters': trainable_params,
            'early_stopped': len(train_losses) < CONFIG['EPOCHS']
        }
    }
    
    with open(f"{CONFIG['RESULTS_DIR']}/training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"Final Results:")
    print(f"  Total epochs: {len(train_losses)}")
    print(f"  Best validation loss: {early_stopping.best_loss:.4f}")
    print(f"  Final training loss: {train_losses[-1]:.4f}")
    print(f"  Final validation loss: {val_losses[-1]:.4f}")
    print(f"  Model saved to: {CONFIG['MODEL_PATH']}")
    print(f"  Training curves saved to: {CONFIG['RESULTS_DIR']}/training_curves.png")
    print(f"  Training summary saved to: {CONFIG['RESULTS_DIR']}/training_summary.json")
    print(f"{'='*80}")
    
    return model, model_info

if __name__ == "__main__":
    trained_model, training_info = train_model()