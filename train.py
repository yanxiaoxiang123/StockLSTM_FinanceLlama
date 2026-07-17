import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import r2_score

from .utils import log_info, get_memory_usage, load_model_weights


MAX_BATCH_ERRORS = 3


def train_stock_model(model, x_train, y_train, x_val, y_val, config, device):
    log_info("Starting stock model training...")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=1e-4,
    )
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        steps_per_epoch=len(train_loader),
        epochs=config['epochs'],
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000,
    )

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        log_info("Using CUDA for training")

    train_losses, val_losses = [], []
    train_maes, val_maes = [], []
    val_r2_scores = []
    best_val_loss = float('inf')
    best_r2 = float('-inf')
    patience_counter = 0
    max_patience = 15

    model_path = config['model_path']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    for epoch in range(config['epochs']):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_mae = 0.0
        successful_batches = 0
        error_count = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for X, y in progress:
            try:
                X = X.to(device)
                y = y.to(device)

                optimizer.zero_grad(set_to_none=True)
                y_pred = model(X, config['prompt'])
                loss = loss_fn(y_pred, y)
                mae = torch.mean(torch.abs(y_pred - y))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                epoch_train_loss += loss.item()
                epoch_train_mae += mae.item()
                successful_batches += 1
                progress.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'mae': f'{mae.item():.6f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                    'mem': f'{get_memory_usage():.1f}MB',
                })
            except Exception as e:
                error_count += 1
                log_info(f"Training step error (occurrence {error_count}): {str(e)[:200]}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if error_count >= MAX_BATCH_ERRORS:
                    log_info(f"Epoch {epoch+1} error limit reached ({MAX_BATCH_ERRORS}), stopping training")
                    raise
                continue

        avg_train_loss = epoch_train_loss / max(successful_batches, 1)
        avg_train_mae = epoch_train_mae / max(successful_batches, 1)

        model.eval()
        epoch_val_loss = 0.0
        epoch_val_mae = 0.0
        val_success = 0
        val_error_count = 0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for X, y in val_loader:
                try:
                    X = X.to(device)
                    y = y.to(device)
                    y_pred = model(X, config['prompt'])
                    loss = loss_fn(y_pred, y)
                    mae = torch.mean(torch.abs(y_pred - y))
                    epoch_val_loss += loss.item()
                    epoch_val_mae += mae.item()
                    val_success += 1
                    val_preds.append(y_pred.cpu().numpy())
                    val_targets.append(y.cpu().numpy())
                except Exception as e:
                    val_error_count += 1
                    log_info(f"Validation step error (occurrence {val_error_count}): {str(e)[:200]}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if val_error_count >= MAX_BATCH_ERRORS:
                        raise
                    continue

        avg_val_loss = epoch_val_loss / max(val_success, 1)
        avg_val_mae = epoch_val_mae / max(val_success, 1)
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        val_r2 = r2_score(val_targets.flatten(), val_preds.flatten())

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_maes.append(avg_train_mae)
        val_maes.append(avg_val_mae)
        val_r2_scores.append(val_r2)

        log_info(
            f"Epoch {epoch+1}/{config['epochs']} - Train loss: {avg_train_loss:.6f}, "
            f"Val loss: {avg_val_loss:.6f}, Train MAE: {avg_train_mae:.6f}, "
            f"Val MAE: {avg_val_mae:.6f}, Val R²: {val_r2:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            log_info(f"Found better validation loss: {best_val_loss:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if val_r2 > best_r2:
            best_r2 = val_r2
            torch.save(model.state_dict(), model_path)
            log_info(f"Saved best model by R², validation R²: {best_r2:.6f}")

        if patience_counter >= max_patience:
            log_info(f"Early stopping triggered at epoch {epoch+1}")
            break

    _plot_training_curves(
        train_losses, val_losses, train_maes, val_maes, val_r2_scores,
        model, val_loader, config, device, model_path)

    return model


def _plot_training_curves(train_losses, val_losses, train_maes, val_maes,
                          val_r2_scores, model, val_loader, config, device,
                          model_path):
    load_model_weights(model, model_path, device)
    model.eval()

    val_preds_final, val_targets_final = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X, config['prompt'])
            val_preds_final.append(y_pred.cpu().numpy())
            val_targets_final.append(y.cpu().numpy())

    val_preds_final = np.vstack(val_preds_final)
    val_targets_final = np.vstack(val_targets_final)

    plt.figure(figsize=(18, 12))

    plt.subplot(2, 3, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()
    plt.title('Training and Validation Loss'); plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 2)
    plt.plot(train_maes, label='Training MAE', color='green')
    plt.plot(val_maes, label='Validation MAE', color='orange')
    plt.xlabel('Epochs'); plt.ylabel('MAE'); plt.legend()
    plt.title('Training and Validation MAE'); plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 3)
    plt.plot(val_r2_scores, label='Validation R²', color='purple')
    plt.xlabel('Epochs'); plt.ylabel('R²'); plt.legend()
    plt.title('Validation R²'); plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 4)
    sample_size = min(500, len(val_preds_final))
    plt.scatter(val_targets_final[:sample_size].flatten(),
                val_preds_final[:sample_size].flatten(),
                alpha=0.6, s=20)
    min_val = min(val_targets_final.min(), val_preds_final.min())
    max_val = max(val_targets_final.max(), val_preds_final.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
             label='Perfect Prediction')
    plt.xlabel('Actual Values'); plt.ylabel('Predicted Values')
    plt.title('Validation Set: Actual vs Predicted'); plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 5)
    errors = val_preds_final.flatten() - val_targets_final.flatten()
    plt.hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Prediction Error'); plt.ylabel('Frequency')
    plt.title('Validation Set: Prediction Error Distribution')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 6)
    sample_size = min(100, len(val_preds_final))
    plt.plot(np.arange(sample_size), val_targets_final[:sample_size, 0],
             label='Actual (Day 1)', linewidth=2, color='blue')
    plt.plot(np.arange(sample_size), val_preds_final[:sample_size, 0],
             label='Predicted (Day 1)', linewidth=2, color='red', linestyle='--')
    plt.xlabel('Sample Index'); plt.ylabel('Stock Price')
    plt.title('Validation Set: Time Series Comparison (Day 1)')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(config['output_dir'], 'training_curves.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_info(f"Training curves saved to: {out_path}")