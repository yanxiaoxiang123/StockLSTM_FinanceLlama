import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .utils import log_info, load_model_weights
from .data import load_single_stock_data, preprocess_stock_data
from .evaluate import calculate_risk_metrics


def predict_future_days(model, X, y, dates, scaler_X, scaler_y, config, device,
                        num_days=5):
    log_info(f"Predicting stock price for next {num_days} days...")

    model = model.to(device)
    load_model_weights(model, config['model_path'], device)
    model.eval()

    X_processed, _, _ = preprocess_stock_data(X, y, fit=True)
    X_scaled = scaler_X.transform(X_processed)

    time_steps = config['time_steps']
    last_sequence = X_scaled[-time_steps:]
    input_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction_scaled = model(input_tensor, config['prompt']).cpu().numpy()

    prediction_original = scaler_y.inverse_transform(prediction_scaled)

    import pandas as pd
    last_date = pd.to_datetime(dates[-1])
    current_date = last_date
    future_dates = []
    for _ in range(num_days):
        current_date = current_date + pd.Timedelta(days=1)
        while current_date.weekday() >= 5:
            current_date = current_date + pd.Timedelta(days=1)
        future_dates.append(current_date)

    future_predictions = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': prediction_original[0],
        'Day': [f'Day {i+1}' for i in range(num_days)],
    })

    last_price = y[-1, 0]
    future_predictions['Price_Change'] = future_predictions['Predicted_Price'] - last_price
    future_predictions['Price_Change_Percent'] = (future_predictions['Price_Change'] / last_price) * 100

    save_dir = config['output_dir']
    os.makedirs(save_dir, exist_ok=True)
    csv_filename = os.path.join(save_dir, 'future_5days_prediction.csv')
    future_predictions.to_csv(csv_filename, index=False)
    log_info(f"Future {num_days}-day prediction saved to: {csv_filename}")

    historical_dates = dates[-30:]
    historical_prices = y[-30:, 0]

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(historical_dates, historical_prices, 'b-', linewidth=2,
             label='Historical Prices (Last 30 days)')
    plt.plot(future_dates, prediction_original[0], 'r--o', linewidth=2,
             markersize=6, label='Future Predictions')
    plt.plot([historical_dates.iloc[-1], future_dates[0]],
             [historical_prices[-1], prediction_original[0][0]],
             'g:', linewidth=1, alpha=0.7)
    plt.xlabel('Date'); plt.ylabel('Stock Price')
    plt.title('Stock Price: Historical vs Future Predictions')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.subplot(2, 1, 2)
    colors = ['green' if c >= 0 else 'red' for c in future_predictions['Price_Change_Percent']]
    bars = plt.bar(future_predictions['Day'], future_predictions['Price_Change_Percent'],
                   color=colors, alpha=0.7)
    for bar, val in zip(bars, future_predictions['Price_Change_Percent']):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + (0.1 if val >= 0 else -0.3),
                 f'{val:.2f}%', ha='center',
                 va='bottom' if val >= 0 else 'top')
    plt.xlabel('Prediction Day'); plt.ylabel('Price Change (%)')
    plt.title('Predicted Price Change from Last Trading Day')
    plt.grid(True, alpha=0.3, axis='y')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(save_dir, 'future_prediction_chart.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()

    log_info("=" * 60)
    log_info(f"Next {num_days} days stock price prediction result:")
    log_info("=" * 60)
    log_info(f"Current price (last trading day): {last_price:.2f}")
    for _, row in future_predictions.iterrows():
        sym = "+" if row['Price_Change'] >= 0 else "-"
        log_info(f"{row['Day']} ({row['Date'].strftime('%Y-%m-%d')}): "
                 f"{row['Predicted_Price']:.2f} ({sym} {row['Price_Change']:+.2f}, "
                 f"{row['Price_Change_Percent']:+.2f}%)")
    log_info("=" * 60)
    return future_predictions


def load_aapl_with_features(file_path):
    try:
        import pandas as pd
        df = pd.read_csv(file_path)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date')

        required_cols = ['open', 'high', 'low', 'close', 'vol']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        from .data import _compute_features, FEATURES
        df = _compute_features(df)

        X = df[FEATURES]
        y = df['close']
        log_info(f"AAPL feature engineering complete, features: {X.shape[1]}, samples: {X.shape[0]}")
        return X.values, y.values.reshape(-1, 1), df['trade_date']
    except Exception as e:
        log_info(f"AAPL feature engineering failed: {e}")
        return None, None, None


def predict_aapl_stock(model, scaler_X, scaler_y, config, device,
                       save_dir='./mg'):
    import pandas as pd
    log_info("Starting AAPL stock prediction...")
    os.makedirs(save_dir, exist_ok=True)

    aapl_file = os.path.join(save_dir, 'aapl.csv')
    if not os.path.exists(aapl_file):
        log_info(f"AAPL data file not found: {aapl_file}")
        return

    aapl_processed, aapl_target, aapl_dates = load_aapl_with_features(aapl_file)
    if aapl_processed is None:
        log_info("AAPL data processing failed")
        return

    log_info(f"AAPL data shape: {aapl_processed.shape}")
    log_info(f"AAPL date range: {aapl_dates.min()} to {aapl_dates.max()}")

    aapl_processed, _, _ = preprocess_stock_data(aapl_processed, aapl_target, fit=True)
    aapl_scaled = scaler_X.transform(aapl_processed)

    time_steps = config['time_steps']
    output_size = config['output_size']
    if len(aapl_scaled) < time_steps + output_size:
        log_info(f"AAPL data insufficient, need at least {time_steps + output_size} days")
        return

    X_sequences, y_sequences, dates_sequences = [], [], []
    for i in range(len(aapl_scaled) - time_steps - output_size + 1):
        X_sequences.append(aapl_scaled[i:i + time_steps])
        y_sequences.append(aapl_target[i + time_steps:i + time_steps + output_size, 0])
        dates_sequences.append(aapl_dates.iloc[i + time_steps:i + time_steps + output_size])

    X_aapl = np.array(X_sequences)
    y_aapl = np.array(y_sequences)
    log_info(f"Created {len(X_sequences)} AAPL prediction sequences")

    X_aapl_tensor = torch.FloatTensor(X_aapl).to(device)

    model = model.to(device)
    load_model_weights(model, config['model_path'], device)
    model.eval()

    all_predictions = []
    batch_size = config['batch_size']
    with torch.no_grad():
        for i in range(0, len(X_aapl_tensor), batch_size):
            batch_X = X_aapl_tensor[i:i + batch_size]
            batch_pred = model(batch_X, config['prompt'])
            all_predictions.append(batch_pred.cpu().numpy())

    aapl_predictions = np.vstack(all_predictions)
    aapl_pred_original = scaler_y.inverse_transform(aapl_predictions)

    log_info(f"AAPL prediction complete, shape: {aapl_pred_original.shape}")

    mae = mean_absolute_error(y_aapl.flatten(), aapl_pred_original.flatten())
    rmse = np.sqrt(mean_squared_error(y_aapl.flatten(), aapl_pred_original.flatten()))
    r2 = r2_score(y_aapl.flatten(), aapl_pred_original.flatten())

    log_info("=" * 60)
    log_info("AAPL prediction evaluation metrics:")
    log_info(f"MAE: {mae:.6f}, RMSE: {rmse:.6f}, R²: {r2:.6f}")
    log_info("=" * 60)

    risk_metrics = calculate_risk_metrics(y_aapl, aapl_pred_original,
                                          risk_free_rate=0.025)

    results_data = []
    for i in range(len(aapl_pred_original)):
        for day in range(output_size):
            results_data.append({
                'Sequence_Index': i,
                'Day': day + 1,
                'Date': dates_sequences[i].iloc[day].strftime('%Y-%m-%d'),
                'Actual_Price': y_aapl[i, day],
                'Predicted_Price': aapl_pred_original[i, day],
                'Absolute_Error': abs(y_aapl[i, day] - aapl_pred_original[i, day]),
                'Relative_Error_Percent': abs(y_aapl[i, day] - aapl_pred_original[i, day]) / y_aapl[i, day] * 100,
            })
    results_df = pd.DataFrame(results_data)
    results_csv = os.path.join(save_dir, 'aapl_prediction_results.csv')
    results_df.to_csv(results_csv, index=False)
    log_info(f"AAPL detailed predictions saved to: {results_csv}")

    summary_df = pd.DataFrame([
        {'Metric': 'MAE', 'Value': mae},
        {'Metric': 'RMSE', 'Value': rmse},
        {'Metric': 'R²', 'Value': r2},
        {'Metric': 'Actual_Volatility', 'Value': risk_metrics['actual_volatility']},
        {'Metric': 'Predicted_Volatility', 'Value': risk_metrics['predicted_volatility']},
        {'Metric': 'Actual_Sharpe_Ratio', 'Value': risk_metrics['actual_sharpe']},
        {'Metric': 'Predicted_Sharpe_Ratio', 'Value': risk_metrics['predicted_sharpe']},
    ])
    summary_csv = os.path.join(save_dir, 'aapl_prediction_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    log_info(f"AAPL prediction summary saved to: {summary_csv}")

    historical_prices = aapl_target[-30:]
    historical_dates = aapl_dates.iloc[-30:]
    create_aapl_visualizations(y_aapl, aapl_pred_original, dates_sequences,
                               risk_metrics, mae, rmse, r2, save_dir,
                               historical_prices, historical_dates)
    log_info("AAPL prediction complete")


def create_aapl_visualizations(y_actual, y_predicted, dates_sequences,
                               risk_metrics, mae, rmse, r2, save_dir,
                               historical_prices, historical_dates):
    log_info("Creating AAPL prediction visualizations...")
    plt.figure(figsize=(20, 16))

    for day in range(5):
        plt.subplot(4, 3, day + 1)
        sample = min(200, len(y_actual))
        plt.plot(y_actual[:sample, day], label='Actual', linewidth=2, alpha=0.8)
        plt.plot(y_predicted[:sample, day], label='Predicted', linewidth=2,
                 linestyle='--', alpha=0.8)
        plt.xlabel('Sample Index'); plt.ylabel('AAPL Stock Price')
        plt.title(f'Day {day+1} AAPL Prediction vs Actual')
        plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(4, 3, 6)
    metrics = ['MAE', 'RMSE', 'R²']
    values = [mae, rmse, r2]
    colors = ['blue', 'orange', 'green']
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.ylabel('Metric Value'); plt.title('AAPL Overall Evaluation Metrics')
    plt.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(values)*0.01,
                 f'{val:.4f}', ha='center', va='bottom')

    plt.subplot(4, 3, 7)
    risk_names = ['Volatility', 'Sharpe Ratio']
    actual_risk = [risk_metrics['actual_volatility'], risk_metrics['actual_sharpe']]
    pred_risk = [risk_metrics['predicted_volatility'], risk_metrics['predicted_sharpe']]
    x = np.arange(len(risk_names)); width = 0.35
    plt.bar(x - width/2, actual_risk, width, label='Actual', alpha=0.8, color='blue')
    plt.bar(x + width/2, pred_risk, width, label='Predicted', alpha=0.8, color='red')
    plt.xticks(x, risk_names); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(4, 3, 8)
    volatility_mae = mean_absolute_error(
        [risk_metrics['actual_volatility']], [risk_metrics['predicted_volatility']])
    sharpe_mae = mean_absolute_error(
        [risk_metrics['actual_sharpe']], [risk_metrics['predicted_sharpe']])
    risk_mae_metrics = ['Volatility MAE', 'Sharpe MAE']
    risk_mae_values = [volatility_mae, sharpe_mae]
    plt.bar(risk_mae_metrics, risk_mae_values, color=['purple', 'brown'], alpha=0.7)
    plt.ylabel('MAE Value'); plt.title('AAPL Risk Metrics Evaluation (MAE)')
    plt.grid(True, alpha=0.3, axis='y'); plt.xticks(rotation=45)
    for i, val in enumerate(risk_mae_values):
        plt.text(i, val + max(risk_mae_values)*0.01, f'{val:.6f}',
                 ha='center', va='bottom')

    plt.subplot(4, 3, 9)
    actual_returns = [(y_actual[i, -1] - y_actual[i, 0]) / y_actual[i, 0]
                      for i in range(len(y_actual))]
    preds_returns = [(y_predicted[i, -1] - y_predicted[i, 0]) / y_predicted[i, 0]
                     for i in range(len(y_predicted))]
    plt.hist(actual_returns, bins=30, alpha=0.7, label='Actual Returns',
             density=True, color='blue')
    plt.hist(preds_returns, bins=30, alpha=0.7, label='Predicted Returns',
             density=True, color='red')
    plt.xlabel('5-Day Returns'); plt.ylabel('Density')
    plt.title('AAPL Returns Distribution Comparison')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(4, 3, 10)
    sample = min(500, len(y_actual))
    plt.scatter(y_actual[:sample].flatten(), y_predicted[:sample].flatten(),
                alpha=0.6, s=20, c='blue')
    min_val = min(y_actual.min(), y_predicted.min())
    max_val = max(y_actual.max(), y_predicted.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
             label='Perfect Prediction')
    plt.xlabel('Actual AAPL Prices'); plt.ylabel('Predicted AAPL Prices')
    plt.title('AAPL Prediction Accuracy Scatter'); plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 3, 11)
    days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5']
    example_idx = min(10, len(y_actual) - 1)
    plt.plot(days, y_actual[example_idx], 'bo-', linewidth=2, markersize=8, label='Actual')
    plt.plot(days, y_predicted[example_idx], 'ro--', linewidth=2, markersize=8, label='Predicted')
    plt.xlabel('Prediction Day'); plt.ylabel('AAPL Stock Price')
    plt.title(f'AAPL 5-Day Prediction Example (Seq {example_idx})')
    plt.legend(); plt.grid(True, alpha=0.3); plt.xticks(rotation=45)

    plt.subplot(4, 3, 12)
    summary_text = (
        f"AAPL Prediction Metrics:\n\nMAE: {mae:.6f}\nRMSE: {rmse:.6f}\nR²: {r2:.6f}\n\n"
        f"Risk Metrics:\nActual Volatility: {risk_metrics['actual_volatility']:.4f}\n"
        f"Predicted Volatility: {risk_metrics['predicted_volatility']:.4f}\n"
        f"Actual Sharpe: {risk_metrics['actual_sharpe']:.4f}\n"
        f"Predicted Sharpe: {risk_metrics['predicted_sharpe']:.4f}\n\n"
        f"Data Statistics:\nPrediction Sequences: {len(y_actual)}\nPrediction Days: 5"
    )
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes,
             fontsize=10, va='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    plt.axis('off')

    plt.tight_layout()
    chart_path = os.path.join(save_dir, 'aapl_prediction_comprehensive_analysis.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_info(f"AAPL prediction visualization saved to: {chart_path}")

    plt.figure(figsize=(12, 8))
    last_idx = len(y_actual) - 1

    plt.subplot(2, 1, 1)
    plt.plot(historical_dates, historical_prices, 'b-', linewidth=2,
             label='Historical Prices (Last 30 days)')
    example_dates = dates_sequences[last_idx]
    plt.plot(example_dates, y_predicted[last_idx], 'r--o', linewidth=2,
             markersize=6, label='AAPL Predictions')
    plt.plot([historical_dates.iloc[-1], example_dates.iloc[0]],
             [historical_prices[-1], y_predicted[last_idx, 0]],
             'g:', linewidth=1, alpha=0.7)
    plt.xlabel('Date'); plt.ylabel('AAPL Stock Price')
    plt.title('AAPL Stock Price: Historical vs Future Predictions')
    plt.legend(); plt.grid(True, alpha=0.3); plt.xticks(rotation=45)

    plt.subplot(2, 1, 2)
    base_price = y_actual[last_idx, 0]
    price_changes = [(p - base_price) / base_price * 100 for p in y_predicted[last_idx]]
    colors = ['green' if c >= 0 else 'red' for c in price_changes]
    bars = plt.bar(days, price_changes, color=colors, alpha=0.7)
    for bar, val in zip(bars, price_changes):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + (0.1 if val >= 0 else -0.3),
                 f'{val:.2f}%', ha='center',
                 va='bottom' if val >= 0 else 'top')
    plt.xlabel('Prediction Day'); plt.ylabel('Price Change (%)')
    plt.title('AAPL Predicted Price Change from Base Day')
    plt.grid(True, alpha=0.3, axis='y')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    plt.tight_layout()
    future_chart = os.path.join(save_dir, 'aapl_future_prediction_chart.png')
    plt.savefig(future_chart, dpi=300, bbox_inches='tight')
    plt.close()
    log_info(f"AAPL future prediction chart saved to: {future_chart}")