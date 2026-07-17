import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .utils import log_info, load_model_weights


def safe_r2_single(actual, predicted):
    if len(actual) < 2:
        return float('nan')
    try:
        return r2_score(actual, predicted)
    except Exception:
        return float('nan')


def calculate_risk_metrics(actual_prices, predicted_prices, risk_free_rate=0.025):
    actual_prices = np.array(actual_prices)
    predicted_prices = np.array(predicted_prices)

    if actual_prices.ndim == 2 and actual_prices.shape[1] > 1:
        actual_returns = (actual_prices[:, -1] - actual_prices[:, 0]) / actual_prices[:, 0]
        predicted_returns = (predicted_prices[:, -1] - predicted_prices[:, 0]) / predicted_prices[:, 0]
    else:
        if actual_prices.ndim == 2:
            actual_prices = actual_prices.flatten()
            predicted_prices = predicted_prices.flatten()
        actual_returns = np.diff(actual_prices) / actual_prices[:-1]
        predicted_returns = np.diff(predicted_prices) / predicted_prices[:-1]

    actual_returns = actual_returns[np.isfinite(actual_returns)]
    predicted_returns = predicted_returns[np.isfinite(predicted_returns)]

    actual_volatility = (np.std(actual_returns, ddof=1) * np.sqrt(252 / 5)
                         if len(actual_returns) > 1 else 0)
    predicted_volatility = (np.std(predicted_returns, ddof=1) * np.sqrt(252 / 5)
                            if len(predicted_returns) > 1 else 0)
    actual_mean_return = np.mean(actual_returns) * (252 / 5) if len(actual_returns) > 0 else 0
    predicted_mean_return = np.mean(predicted_returns) * (252 / 5) if len(predicted_returns) > 0 else 0

    actual_sharpe = ((actual_mean_return - risk_free_rate) / actual_volatility
                     if actual_volatility != 0 else 0)
    predicted_sharpe = ((predicted_mean_return - risk_free_rate) / predicted_volatility
                        if predicted_volatility != 0 else 0)

    return {
        'actual_volatility': actual_volatility,
        'predicted_volatility': predicted_volatility,
        'actual_sharpe': actual_sharpe,
        'predicted_sharpe': predicted_sharpe,
        'actual_returns': actual_returns,
        'predicted_returns': predicted_returns,
    }


def evaluate_stock_model(model, x_test, y_test, scaler_y, config, device):
    log_info("Starting stock model evaluation...")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = model.to(device)
    load_model_weights(model, config['model_path'], device)
    model.eval()

    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=config['batch_size'],
    )

    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X, config['prompt'])
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    test_pred_np = np.vstack(all_preds)
    y_test_np = np.vstack(all_targets)

    mse = mean_squared_error(y_test_np.flatten(), test_pred_np.flatten())
    mae = mean_absolute_error(y_test_np.flatten(), test_pred_np.flatten())
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_np.flatten(), test_pred_np.flatten())

    log_info("Test set basic evaluation metrics:")
    log_info(f" - MAE  : {mae:.6f}")
    log_info(f" - RMSE : {rmse:.6f}")
    log_info(f" - R²   : {r2:.6f}")

    test_pred_inverse = scaler_y.inverse_transform(test_pred_np)
    y_test_inverse = scaler_y.inverse_transform(y_test_np)

    log_info("Calculating risk metrics...")
    risk_metrics = calculate_risk_metrics(y_test_inverse, test_pred_inverse,
                                          risk_free_rate=0.025)

    volatility_mae = mean_absolute_error(
        [risk_metrics['actual_volatility']], [risk_metrics['predicted_volatility']])
    volatility_rmse = np.sqrt(mean_squared_error(
        [risk_metrics['actual_volatility']], [risk_metrics['predicted_volatility']]))
    volatility_r2 = safe_r2_single(
        [risk_metrics['actual_volatility']], [risk_metrics['predicted_volatility']])

    sharpe_mae = mean_absolute_error(
        [risk_metrics['actual_sharpe']], [risk_metrics['predicted_sharpe']])
    sharpe_rmse = np.sqrt(mean_squared_error(
        [risk_metrics['actual_sharpe']], [risk_metrics['predicted_sharpe']]))
    sharpe_r2 = safe_r2_single(
        [risk_metrics['actual_sharpe']], [risk_metrics['predicted_sharpe']])

    def _fmt(r2_val):
        return f"{r2_val:.6f}" if not np.isnan(r2_val) else "N/A"

    log_info("Risk metrics evaluation:")
    log_info(f"Volatility - Actual: {risk_metrics['actual_volatility']:.4f}, "
             f"Predicted: {risk_metrics['predicted_volatility']:.4f}")
    log_info(f"Volatility evaluation - MAE: {volatility_mae:.6f}, RMSE: {volatility_rmse:.6f}, "
             f"R²: {_fmt(volatility_r2)}")
    log_info(f"Sharpe ratio - Actual: {risk_metrics['actual_sharpe']:.4f}, "
             f"Predicted: {risk_metrics['predicted_sharpe']:.4f}")
    log_info(f"Sharpe ratio evaluation - MAE: {sharpe_mae:.6f}, RMSE: {sharpe_rmse:.6f}, "
             f"R²: {_fmt(sharpe_r2)}")

    _save_evaluation_results(y_test_inverse, test_pred_inverse, risk_metrics,
                             mae, rmse, r2, volatility_mae, volatility_rmse,
                             volatility_r2, sharpe_mae, sharpe_rmse, sharpe_r2,
                             config)
    return r2


def _save_evaluation_results(y_test_inverse, test_pred_inverse, risk_metrics,
                              mae, rmse, r2, vol_mae, vol_rmse, vol_r2,
                              sh_mae, sh_rmse, sh_r2, config):
    import pandas as pd
    save_dir = config['output_dir']

    results_df = pd.DataFrame({
        'Sample_Index': np.arange(len(y_test_inverse)),
    })
    for day in range(5):
        results_df[f'Day{day+1}_Actual'] = y_test_inverse[:, day]
        results_df[f'Day{day+1}_Predicted'] = test_pred_inverse[:, day]
    for day in range(1, 6):
        results_df[f'Day{day}_Error'] = np.abs(
            results_df[f'Day{day}_Actual'] - results_df[f'Day{day}_Predicted'])
        results_df[f'Day{day}_Error_Percent'] = (
            results_df[f'Day{day}_Error'] / results_df[f'Day{day}_Actual']) * 100

    csv_path = os.path.join(save_dir, 'stock_prediction_results.csv')
    results_df.to_csv(csv_path, index=False)
    log_info(f"Prediction results saved to: {csv_path}")

    plt.figure(figsize=(20, 15))
    for day in range(5):
        plt.subplot(3, 3, day + 1)
        sample = min(100, len(test_pred_inverse))
        plt.plot(y_test_inverse[:sample, day], label='Actual', linewidth=2)
        plt.plot(test_pred_inverse[:sample, day], label='Predicted',
                 linewidth=2, linestyle='--')
        plt.xlabel('Sample Index'); plt.ylabel('Stock Price')
        plt.title(f'Day {day+1} Prediction'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(3, 3, 6)
    plt.text(0.5, 0.7, f'Overall Metrics:\nMAE: {mae:.6f}\nRMSE: {rmse:.6f}\nR²: {r2:.6f}',
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    plt.axis('off')

    plt.subplot(3, 3, 7)
    labels = ['Volatility', 'Sharpe Ratio']
    actual_risk = [risk_metrics['actual_volatility'], risk_metrics['actual_sharpe']]
    pred_risk = [risk_metrics['predicted_volatility'], risk_metrics['predicted_sharpe']]
    x = np.arange(len(labels)); width = 0.35
    plt.bar(x - width/2, actual_risk, width, label='Actual', alpha=0.8)
    plt.bar(x + width/2, pred_risk, width, label='Predicted', alpha=0.8)
    plt.xlabel('Risk Metrics'); plt.ylabel('Value')
    plt.title('Risk Metrics Comparison'); plt.xticks(x, labels)
    plt.legend(); plt.grid(True, alpha=0.3)

    def _fmt(v):
        return f"{v:.6f}" if not np.isnan(v) else "N/A"
    plt.subplot(3, 3, 8)
    plt.text(0.5, 0.7,
             f'Risk Metrics Evaluation:\n\nVolatility:\nMAE: {vol_mae:.6f}\n'
             f'RMSE: {vol_rmse:.6f}\nR²: {_fmt(vol_r2)}\n\nSharpe Ratio:\n'
             f'MAE: {sh_mae:.6f}\nRMSE: {sh_rmse:.6f}\nR²: {_fmt(sh_r2)}',
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    plt.axis('off')

    plt.subplot(3, 3, 9)
    plt.hist(risk_metrics['actual_returns'], bins=30, alpha=0.7,
             label='Actual Returns', density=True)
    plt.hist(risk_metrics['predicted_returns'], bins=30, alpha=0.7,
             label='Predicted Returns', density=True)
    plt.xlabel('Daily Returns'); plt.ylabel('Density')
    plt.title('Returns Distribution Comparison'); plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(save_dir, 'prediction_results.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_info(f"Evaluation chart saved to: {png_path}")