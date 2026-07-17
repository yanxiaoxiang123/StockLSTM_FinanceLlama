import os
import torch

from .config import get_default_config, ensure_dirs
from .utils import log_info
from .data import load_all_stock_data, create_stock_data_split, create_single_stock_sequences
from .model import StockLSTM_FinanceLlama
from .train import train_stock_model
from .evaluate import evaluate_stock_model
from .backtest import backtest_model, create_backtest_visualizations
from .predict import predict_future_days, predict_aapl_stock


def main():
    log_info("Stock prediction program started...")

    config = get_default_config()
    device = config['device']
    log_info(f"Using device: {device}")

    ensure_dirs(config)

    log_info("Loading stock data...")
    stock_data_list = load_all_stock_data(config['data_dir'])
    if stock_data_list is None:
        log_info("Data loading failed, exiting")
        return

    (x_train, y_train, x_val, y_val, x_test, y_test,
     scaler_X, scaler_y) = create_stock_data_split(
        stock_data_list, config['time_steps'], config['output_size'],
        save_dir=config['output_dir'])

    config['input_size'] = x_train.shape[2]
    log_info(f"Input size after feature engineering: {config['input_size']}")

    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    model = StockLSTM_FinanceLlama(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        output_size=config['output_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        llama_model_path=config['llama_model_path'],
    ).to(device)
    model.clear_finance_llama_cache()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trained_model = train_stock_model(
        model, x_train, y_train, x_val, y_val, config, device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    test_r2 = evaluate_stock_model(
        trained_model, x_test, y_test, scaler_y, config, device)
    log_info(f"Final test set R² score: {test_r2:.6f}")

    single_stock_path = os.path.join(config['data_dir'], '000001.sz.csv')
    log_info(f"Building single stock backtest data: {single_stock_path}")
    x_backtest, y_backtest = create_single_stock_sequences(
        single_stock_path, scaler_X, scaler_y,
        config['time_steps'], config['output_size'])

    if x_backtest is None or y_backtest is None:
        log_info("Single stock backtest data construction failed, falling back to test set")
        x_backtest, y_backtest = x_test, y_test
    else:
        x_backtest = x_backtest.to(device)
        y_backtest = y_backtest.to(device)

    backtest_results = backtest_model(
        trained_model, x_backtest, y_backtest, scaler_y, config, device,
        save_dir='./backtest_results')
    create_backtest_visualizations(backtest_results, save_dir='./backtest_results')

    last_stock = stock_data_list[-1]
    last_X = last_stock['X']
    last_y = last_stock['y']
    last_dates = last_stock['dates']
    future_predictions = predict_future_days(
        trained_model, last_X, last_y, last_dates,
        scaler_X, scaler_y, config, device, num_days=5)

    log_info("Starting AAPL stock prediction...")
    predict_aapl_stock(trained_model, scaler_X, scaler_y, config, device,
                       save_dir='./mg')

    log_info("Stock prediction program finished")


if __name__ == "__main__":
    main()