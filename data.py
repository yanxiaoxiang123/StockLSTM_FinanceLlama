import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
import pickle
from .utils import log_info


FEATURES = [
    'open', 'high', 'low', 'volatility_20', 'daily_range',
    'volume_change', 'macd', 'rsi', 'ma5', 'ma20', 'ema12', 'ema26',
    'momentum', 'vol_ma5', 'atr', 'obv', 'bollinger_upper', 'bollinger_lower',
    'price_volume_ratio',
]


def _compute_features(df):
    df = df.sort_values('trade_date')

    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['volatility_20'] = df['close'].pct_change().rolling(20).std()
    df['daily_range'] = (df['high'] - df['low']) / df['low']
    df['volume_change'] = df['vol'].pct_change()

    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))

    df['ema12'] = ema12
    df['ema26'] = ema26
    df['momentum'] = df['close'].pct_change(5)
    df['vol_ma5'] = df['vol'].rolling(5).mean()

    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    df['obv'] = (np.sign(df['close'].diff()) * df['vol']).cumsum()

    df['bollinger_upper'] = df['ma20'] + 2 * df['close'].rolling(20).std()
    df['bollinger_lower'] = df['ma20'] - 2 * df['close'].rolling(20).std()
    df['price_volume_ratio'] = df['close'] / df['vol']

    df = df.dropna()
    return df


def load_single_stock_data(file_path):
    try:
        df = pd.read_csv(file_path)
        date_fmt = '%Y%m%d'
        try:
            df['trade_date'] = pd.to_datetime(df['trade_date'], format=date_fmt)
        except (ValueError, TypeError):
            df['trade_date'] = pd.to_datetime(df['trade_date'])

        df = _compute_features(df)

        X = df[FEATURES]
        y = df['close']
        return X.values, y.values.reshape(-1, 1), df['trade_date']
    except Exception as e:
        log_info(f"Failed to load file {file_path}: {e}")
        return None, None, None


def load_all_stock_data(data_dir='./data'):
    try:
        log_info(f"Loading stock data directory: {data_dir}")
        csv_files = sorted(
            f for f in os.listdir(data_dir) if f.endswith('.csv')
        )
        log_info(f"Found {len(csv_files)} CSV files")

        stock_data_list = []
        for file_name in csv_files:
            file_path = os.path.join(data_dir, file_name)
            log_info(f"Processing: {file_name}")
            X, y, dates = load_single_stock_data(file_path)
            if X is None:
                continue
            stock_data_list.append({
                'stock_id': file_name,
                'X': X,
                'y': y,
                'dates': dates,
            })

        if not stock_data_list:
            log_info("No stock data loaded successfully")
            return None

        log_info(f"Data loading complete, {len(stock_data_list)} stocks total")
        return stock_data_list

    except Exception as e:
        log_info(f"Data loading failed: {e}")
        return None


def create_sequences_per_stock(X, y, dates, time_steps, output_size):
    n = len(X)
    m = n - time_steps - output_size + 1
    if m <= 0:
        return None, None, None

    X_seq = np.array([X[i:i + time_steps] for i in range(m)])
    y_seq = np.array([y[i + time_steps:i + time_steps + output_size].flatten()
                      for i in range(m)])
    target_dates = dates.iloc[time_steps:time_steps + m].values

    return X_seq, y_seq, target_dates


def split_by_global_time(all_samples, train_ratio=0.7, val_ratio=0.15):
    if not all_samples:
        raise ValueError("No samples to split")

    X_all = np.vstack([s['X_seq'] for s in all_samples])
    y_all = np.vstack([s['y_seq'] for s in all_samples])
    t_all = np.concatenate([s['target_dates'] for s in all_samples])

    order = np.argsort(t_all)
    X_all = X_all[order]
    y_all = y_all[order]
    t_sorted = t_all[order]

    n = len(X_all)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    x_train, y_train = X_all[:train_end], y_all[:train_end]
    x_val, y_val = X_all[train_end:val_end], y_all[train_end:val_end]
    x_test, y_test = X_all[val_end:], y_all[val_end:]

    log_info(
        f"Global time split: train {x_train.shape} (target up to {t_sorted[min(train_end-1, n-1)]}), "
        f"val {x_val.shape}, test {x_test.shape} (target from {t_sorted[min(val_end, n-1)]})"
    )
    return x_train, y_train, x_val, y_val, x_test, y_test


def preprocess_stock_data(X, y, fit=True, bounds=None):
    original_shape = X.shape
    X_flat = X.reshape(-1, X.shape[-1])
    y_flat = y.reshape(-1, y.shape[-1]) if y.ndim > 1 else y.reshape(-1, 1)

    X_flat = np.nan_to_num(X_flat, nan=0.0,
                           posinf=np.finfo(np.float64).max,
                           neginf=np.finfo(np.float64).min)
    y_flat = np.nan_to_num(y_flat, nan=0.0,
                           posinf=np.finfo(np.float64).max,
                           neginf=np.finfo(np.float64).min)

    if fit:
        bounds = []
        for col in range(X_flat.shape[1]):
            col_data = X_flat[:, col]
            q1 = np.percentile(col_data, 25)
            q3 = np.percentile(col_data, 75)
            iqr = q3 - q1
            lower = q1 - 3 * iqr
            upper = q3 + 3 * iqr
            bounds.append((lower, upper))
            X_flat[:, col] = np.clip(col_data, lower, upper)
    else:
        for col, (lower, upper) in enumerate(bounds):
            X_flat[:, col] = np.clip(X_flat[:, col], lower, upper)

    X_out = X_flat.reshape(original_shape)
    return X_out, y_flat, bounds


def fit_scalers_on_train(X_train, y_train):
    n_feat = X_train.shape[-1]
    X_train_2d = X_train.reshape(-1, n_feat)
    scaler_X = RobustScaler().fit(X_train_2d)
    scaler_y = RobustScaler().fit(y_train.reshape(-1, 1))
    return scaler_X, scaler_y


def apply_scalers(X, scaler_X, y, scaler_y):
    original_shape = X.shape
    X_2d = X.reshape(-1, original_shape[-1])
    X_scaled = scaler_X.transform(X_2d).reshape(original_shape)
    y_scaled = scaler_y.transform(y.reshape(-1, 1)).reshape(y.shape)
    return X_scaled, y_scaled


def create_stock_data_split(stock_data_list, time_steps, output_size,
                            train_ratio=0.7, val_ratio=0.15,
                            save_dir='./test_output'):
    import torch

    log_info(f"Processing stock data, time_steps: {time_steps}, output_size: {output_size}")
    start_time = __import__('time').time()

    all_samples = []
    for s in tqdm(stock_data_list, desc="Creating sequences per stock"):
        X_seq, y_seq, target_dates = create_sequences_per_stock(
            s['X'], s['y'], s['dates'], time_steps, output_size)
        if X_seq is None:
            continue
        all_samples.append({
            'X_seq': X_seq, 'y_seq': y_seq, 'target_dates': target_dates
        })

    if not all_samples:
        raise RuntimeError("Failed to build any sequence samples")

    x_train, y_train, x_val, y_val, x_test, y_test = split_by_global_time(
        all_samples, train_ratio, val_ratio)

    x_train, y_train, bounds = preprocess_stock_data(x_train, y_train, fit=True)
    x_val, y_val, _ = preprocess_stock_data(x_val, y_val, fit=False, bounds=bounds)
    x_test, y_test, _ = preprocess_stock_data(x_test, y_test, fit=False, bounds=bounds)

    scaler_X, scaler_y = fit_scalers_on_train(x_train, y_train)

    x_train, y_train = apply_scalers(x_train, scaler_X, y_train, scaler_y)
    x_val, y_val = apply_scalers(x_val, scaler_X, y_val, scaler_y)
    x_test, y_test = apply_scalers(x_test, scaler_X, y_test, scaler_y)

    log_info("Data standardization complete (scaler fit on training set only)")

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'scaler_X.pkl'), 'wb') as f:
        pickle.dump(scaler_X, f)
    with open(os.path.join(save_dir, 'scaler_y.pkl'), 'wb') as f:
        pickle.dump(scaler_y, f)
    log_info("Scaler objects saved")

    end_time = __import__('time').time()
    log_info(f"Data processing complete, elapsed: {end_time - start_time:.2f}s")
    log_info(f"Training set: {x_train.shape}, Validation set: {x_val.shape}, Test set: {x_test.shape}")

    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    return (to_t(x_train), to_t(y_train),
            to_t(x_val), to_t(y_val),
            to_t(x_test), to_t(y_test),
            scaler_X, scaler_y)


def create_single_stock_sequences(file_path, scaler_X, scaler_y,
                                  time_steps, output_size,
                                  train_bounds=None):
    log_info(f"Creating sequences for single stock: {file_path}")
    X_raw, y_raw, dates = load_single_stock_data(file_path)
    if X_raw is None:
        log_info("Single stock data loading failed")
        return None, None

    X_seq, y_seq, target_dates = create_sequences_per_stock(
        X_raw, y_raw, dates, time_steps, output_size)
    if X_seq is None:
        log_info("Single stock data insufficient for sequence construction")
        return None, None

    if train_bounds is not None:
        X_seq, y_seq, _ = preprocess_stock_data(
            X_seq, y_seq, fit=False, bounds=train_bounds)
    else:
        X_seq, y_seq, _ = preprocess_stock_data(X_seq, y_seq, fit=True)

    X_scaled, y_scaled = apply_scalers(X_seq, scaler_X, y_seq, scaler_y)

    import torch
    return (torch.tensor(X_scaled, dtype=torch.float32),
            torch.tensor(y_scaled, dtype=torch.float32))