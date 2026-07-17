import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .utils import log_info, load_model_weights
from .data import create_single_stock_sequences


def backtest_model(model, x_test, y_test, scaler_y, config, device,
                   save_dir='./backtest_results'):
    log_info("Starting model backtest...")
    os.makedirs(save_dir, exist_ok=True)

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

    test_pred_inverse = scaler_y.inverse_transform(test_pred_np)
    y_test_inverse = scaler_y.inverse_transform(y_test_np)

    mse = mean_squared_error(y_test_np.flatten(), test_pred_np.flatten())
    mae = mean_absolute_error(y_test_np.flatten(), test_pred_np.flatten())
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_np.flatten(), test_pred_np.flatten())

    from .evaluate import calculate_risk_metrics
    log_info("Calculating backtest risk metrics...")
    risk_metrics = calculate_risk_metrics(
        y_test_inverse, test_pred_inverse, risk_free_rate=0.025)

    volatility_mae = mean_absolute_error(
        [risk_metrics['actual_volatility']], [risk_metrics['predicted_volatility']])
    sharpe_mae = mean_absolute_error(
        [risk_metrics['actual_sharpe']], [risk_metrics['predicted_sharpe']])

    daily_metrics = {}
    for day in range(5):
        day_actual = y_test_inverse[:, day]
        day_pred = test_pred_inverse[:, day]
        daily_metrics[f'day_{day+1}'] = {
            'mae': mean_absolute_error(day_actual, day_pred),
            'rmse': np.sqrt(mean_squared_error(day_actual, day_pred)),
            'r2': r2_score(day_actual, day_pred),
            'mape': np.mean(np.abs((day_actual - day_pred) / day_actual)) * 100,
        }

    actual_returns = []
    predicted_returns = []
    for i in range(len(y_test_inverse)):
        actual_returns.append((y_test_inverse[i, -1] - y_test_inverse[i, 0]) / y_test_inverse[i, 0])
        predicted_returns.append((test_pred_inverse[i, -1] - test_pred_inverse[i, 0]) / test_pred_inverse[i, 0])
    actual_returns = np.array(actual_returns)
    predicted_returns = np.array(predicted_returns)

    stop_loss_pct = 0.03
    take_profit_pct = 0.10
    transaction_cost = 0.001
    slippage = 0.0005
    entry_long_threshold = 0.02
    entry_short_threshold = 0.02
    max_position_size = 0.3
    confidence_scale = 0.10

    portfolio_value = 100000
    current_position = 0
    position_entry_price = 0
    cash = portfolio_value
    strategy_returns = []
    portfolio_values = []
    positions = []
    trades = []

    for i in range(len(y_test_inverse)):
        actual_prices = y_test_inverse[i]
        predicted_prices = test_pred_inverse[i]

        predicted_cum_return = (predicted_prices[-1] - predicted_prices[0]) / predicted_prices[0]
        confidence = min(1.0, abs(predicted_cum_return) / confidence_scale) if confidence_scale > 0 else 0.0

        signal = 0
        if predicted_cum_return > entry_long_threshold:
            signal = 1
        elif predicted_cum_return < -entry_short_threshold:
            signal = -1

        current_price = actual_prices[0]

        if current_position != 0:
            price_change = (current_price - position_entry_price) / position_entry_price
            if current_position > 0:
                if price_change <= -stop_loss_pct or price_change >= take_profit_pct:
                    signal = -1
            else:
                if price_change >= stop_loss_pct or price_change <= -take_profit_pct:
                    signal = 1

        if signal != 0:
            if current_position == 0:
                position_size = int(min(max_position_size, confidence) * cash / current_price)
                if signal == 1 and position_size > 0:
                    cost = position_size * current_price * (1 + transaction_cost + slippage)
                    if cost <= cash:
                        current_position = position_size
                        position_entry_price = current_price
                        cash -= cost
                        trades.append({'type': 'BUY', 'price': current_price,
                                       'size': position_size, 'cost': cost})
                elif signal == -1 and position_size > 0:
                    proceeds = position_size * current_price * (1 - transaction_cost - slippage)
                    current_position = -position_size
                    position_entry_price = current_price
                    cash += proceeds
                    trades.append({'type': 'SHORT', 'price': current_price,
                                   'size': position_size, 'proceeds': proceeds})
            else:
                if (signal == -1 and current_position > 0) or (signal == 1 and current_position < 0):
                    if current_position > 0:
                        proceeds = current_position * current_price * (1 - transaction_cost - slippage)
                        cash += proceeds
                        trades.append({'type': 'SELL', 'price': current_price,
                                       'size': current_position, 'proceeds': proceeds})
                    else:
                        cost = abs(current_position) * current_price * (1 + transaction_cost + slippage)
                        cash -= cost
                        trades.append({'type': 'COVER', 'price': current_price,
                                       'size': abs(current_position), 'cost': cost})
                    current_position = 0
                    position_entry_price = 0

        if current_position > 0:
            portfolio_value = cash + current_position * current_price
        elif current_position < 0:
            portfolio_value = cash - abs(current_position) * current_price
        else:
            portfolio_value = cash

        portfolio_values.append(portfolio_value)
        positions.append(current_position)
        period_return = 0 if i == 0 else (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
        strategy_returns.append(period_return)

    strategy_returns = np.array(strategy_returns)
    portfolio_values = np.array(portfolio_values)

    strategy_total_return = (portfolio_values[-1] - 100000) / 100000
    strategy_sharpe = (np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
                       if np.std(strategy_returns) != 0 else 0)
    win_rate = np.sum(strategy_returns > 0) / len(strategy_returns)
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = np.min(drawdown)
    num_trades = len(trades)
    avg_trade_return = (np.mean([t.get('proceeds', t.get('cost', 0)) for t in trades])
                        if trades else 0)

    backtest_results = {
        'basic_metrics': {'mae': mae, 'rmse': rmse, 'r2': r2, 'mse': mse},
        'risk_metrics': {
            'volatility_mae': volatility_mae, 'sharpe_mae': sharpe_mae,
            'actual_volatility': risk_metrics['actual_volatility'],
            'predicted_volatility': risk_metrics['predicted_volatility'],
            'actual_sharpe': risk_metrics['actual_sharpe'],
            'predicted_sharpe': risk_metrics['predicted_sharpe'],
        },
        'daily_metrics': daily_metrics,
        'strategy_performance': {
            'total_return': strategy_total_return,
            'sharpe_ratio': strategy_sharpe,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'avg_trade_return': avg_trade_return,
            'final_portfolio_value': portfolio_values[-1] if len(portfolio_values) > 0 else 100000,
        },
        'raw_data': {
            'actual_prices': y_test_inverse,
            'predicted_prices': test_pred_inverse,
            'actual_returns': actual_returns,
            'predicted_returns': predicted_returns,
            'strategy_returns': strategy_returns,
            'portfolio_values': portfolio_values,
            'positions': positions,
            'trades': trades,
        },
    }

    _log_backtest_summary(backtest_results, stop_loss_pct, take_profit_pct,
                          transaction_cost, slippage, max_position_size)
    _save_backtest_results(backtest_results, y_test_inverse, test_pred_inverse,
                           actual_returns, predicted_returns,
                           strategy_returns, portfolio_values, positions,
                           save_dir)
    create_backtest_visualizations(backtest_results, save_dir)
    return backtest_results


def _log_backtest_summary(br, stop_loss_pct, take_profit_pct,
                          transaction_cost, slippage, max_position_size):
    bm = br['basic_metrics']; rm = br['risk_metrics']; sp = br['strategy_performance']
    log_info("=" * 60)
    log_info("Optimized backtest results summary:")
    log_info("=" * 60)
    log_info(f"Basic metrics: MAE: {bm['mae']:.6f}, RMSE: {bm['rmse']:.6f}, R²: {bm['r2']:.6f}")
    log_info(f"Risk metrics:")
    log_info(f"  Volatility MAE: {rm['volatility_mae']:.6f}, Sharpe MAE: {rm['sharpe_mae']:.6f}")
    log_info(f"  Actual volatility: {rm['actual_volatility']:.4f}, Predicted volatility: {rm['predicted_volatility']:.4f}")
    log_info(f"  Actual Sharpe: {rm['actual_sharpe']:.4f}, Predicted Sharpe: {rm['predicted_sharpe']:.4f}")
    log_info(f"Strategy performance:")
    log_info(f"  Total return: {sp['total_return']*100:.2f}%")
    log_info(f"  Strategy Sharpe: {sp['sharpe_ratio']:.4f}")
    log_info(f"  Win rate: {sp['win_rate']:.2%}")
    log_info(f"  Max drawdown: {sp['max_drawdown']*100:.2f}%")
    log_info(f"  Trades: {sp['num_trades']}, Avg trade return: {sp['avg_trade_return']:.2f}")
    log_info(f"  Final portfolio value: {sp['final_portfolio_value']:.2f}")
    log_info(f"Parameters: stop {stop_loss_pct*100:.1f}%, take {take_profit_pct*100:.1f}%, "
             f"cost {transaction_cost*100:.2f}%, slippage {slippage*100:.3f}%, "
             f"max position {max_position_size*100:.1f}%")
    log_info("=" * 60)


def _save_backtest_results(br, y_test_inverse, test_pred_inverse,
                           actual_returns, predicted_returns,
                           strategy_returns, portfolio_values, positions,
                           save_dir):
    import pandas as pd
    results_df = pd.DataFrame({'Sample_Index': np.arange(len(y_test_inverse))})
    for day in range(5):
        results_df[f'Day{day+1}_Actual'] = y_test_inverse[:, day]
        results_df[f'Day{day+1}_Predicted'] = test_pred_inverse[:, day]
    results_df['Actual_5Day_Return'] = actual_returns
    results_df['Predicted_5Day_Return'] = predicted_returns
    results_df['Strategy_Return'] = strategy_returns
    results_df['Portfolio_Value'] = portfolio_values
    results_df['Position'] = positions
    for day in range(1, 6):
        results_df[f'Day{day}_Error'] = np.abs(
            results_df[f'Day{day}_Actual'] - results_df[f'Day{day}_Predicted'])
        results_df[f'Day{day}_Error_Percent'] = (
            results_df[f'Day{day}_Error'] / results_df[f'Day{day}_Actual']) * 100

    csv_path = os.path.join(save_dir, 'backtest_detailed_results.csv')
    results_df.to_csv(csv_path, index=False)
    log_info(f"Detailed backtest results saved to: {csv_path}")

    sp = br['strategy_performance']; bm = br['basic_metrics']; rm = br['risk_metrics']
    summary_df = pd.DataFrame([
        {'Metric': 'MAE', 'Value': bm['mae']},
        {'Metric': 'RMSE', 'Value': bm['rmse']},
        {'Metric': 'R²', 'Value': bm['r2']},
        {'Metric': 'Volatility_MAE', 'Value': rm['volatility_mae']},
        {'Metric': 'Sharpe_MAE', 'Value': rm['sharpe_mae']},
        {'Metric': 'Strategy_Total_Return', 'Value': sp['total_return']},
        {'Metric': 'Strategy_Sharpe_Ratio', 'Value': sp['sharpe_ratio']},
        {'Metric': 'Win_Rate', 'Value': sp['win_rate']},
        {'Metric': 'Max_Drawdown', 'Value': sp['max_drawdown']},
        {'Metric': 'Num_Trades', 'Value': sp['num_trades']},
        {'Metric': 'Avg_Trade_Return', 'Value': sp['avg_trade_return']},
        {'Metric': 'Final_Portfolio_Value', 'Value': sp['final_portfolio_value']},
    ])
    summary_path = os.path.join(save_dir, 'backtest_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    log_info(f"Backtest summary saved to: {summary_path}")


def create_backtest_visualizations(backtest_results, save_dir='./backtest_results'):
    log_info("Creating backtest visualizations...")
    rd = backtest_results['raw_data']
    actual_prices = rd['actual_prices']
    predicted_prices = rd['predicted_prices']
    actual_returns = rd['actual_returns']
    predicted_returns = rd['predicted_returns']
    strategy_returns = rd['strategy_returns']

    plt.figure(figsize=(20, 16))

    for day in range(5):
        plt.subplot(4, 3, day + 1)
        sample = min(200, len(actual_prices))
        plt.plot(actual_prices[:sample, day], label='Actual', linewidth=2, alpha=0.8)
        plt.plot(predicted_prices[:sample, day], label='Predicted', linewidth=2,
                 linestyle='--', alpha=0.8)
        plt.xlabel('Sample Index'); plt.ylabel('Stock Price')
        plt.title(f'Day {day+1} Prediction vs Actual')
        plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(4, 3, 6)
    rm = backtest_results['risk_metrics']
    labels = ['Volatility', 'Sharpe Ratio']
    actual_v = [rm['actual_volatility'], rm['actual_sharpe']]
    pred_v = [rm['predicted_volatility'], rm['predicted_sharpe']]
    x = np.arange(len(labels)); width = 0.35
    plt.bar(x - width/2, actual_v, width, label='Actual', alpha=0.8, color='blue')
    plt.bar(x + width/2, pred_v, width, label='Predicted', alpha=0.8, color='red')
    plt.xticks(x, labels); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(4, 3, 7)
    plt.hist(actual_returns, bins=30, alpha=0.7, label='Actual Returns',
             density=True, color='blue')
    plt.hist(predicted_returns, bins=30, alpha=0.7, label='Predicted Returns',
             density=True, color='red')
    plt.xlabel('5-Day Returns'); plt.ylabel('Density')
    plt.title('Returns Distribution Comparison'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(4, 3, 8)
    plt.hist(strategy_returns, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Strategy Returns'); plt.ylabel('Frequency')
    plt.title('Strategy Returns Distribution')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(4, 3, 9)
    plt.plot(np.cumsum(strategy_returns), linewidth=2, color='green')
    plt.xlabel('Trade Number'); plt.ylabel('Cumulative Return')
    plt.title('Cumulative Strategy Returns'); plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    plt.subplot(4, 3, 10)
    dm = backtest_results['daily_metrics']
    days = [f'Day {i+1}' for i in range(5)]
    mae_values = [dm[f'day_{i+1}']['mae'] for i in range(5)]
    r2_values = [dm[f'day_{i+1}']['r2'] for i in range(5)]
    x = np.arange(len(days)); width = 0.35
    ax1 = plt.gca(); ax2 = ax1.twinx()
    ax1.bar(x - width/2, mae_values, width, label='MAE', alpha=0.8, color='orange')
    ax2.bar(x + width/2, r2_values, width, label='R²', alpha=0.8, color='purple')
    ax1.set_xticks(x); ax1.set_xticklabels(days); ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.subplot(4, 3, 11)
    bm = backtest_results['basic_metrics']; sp = backtest_results['strategy_performance']
    summary_text = (
        f"MAE: {bm['mae']:.6f}\nRMSE: {bm['rmse']:.6f}\nR²: {bm['r2']:.6f}\n"
        f"Volatility MAE: {rm['volatility_mae']:.6f}\nSharpe MAE: {rm['sharpe_mae']:.6f}\n"
        f"Total Return: {sp['total_return']:.4f}\nSharpe Ratio: {sp['sharpe_ratio']:.4f}\n"
        f"Win Rate: {sp['win_rate']:.2%}\nTrades: {sp['num_trades']}"
    )
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes,
             fontsize=10, va='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    plt.axis('off')

    plt.subplot(4, 3, 12)
    sample = min(500, len(actual_prices))
    plt.scatter(actual_prices[:sample].flatten(), predicted_prices[:sample].flatten(),
                alpha=0.6, s=20, c='blue')
    min_val = min(actual_prices.min(), predicted_prices.min())
    max_val = max(actual_prices.max(), predicted_prices.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
             label='Perfect Prediction')
    plt.xlabel('Actual Prices'); plt.ylabel('Predicted Prices')
    plt.title('Prediction Accuracy Scatter'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(save_dir, 'backtest_comprehensive_analysis.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_info(f"Backtest visualization saved to: {out_path}")