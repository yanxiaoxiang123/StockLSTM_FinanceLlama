"""Quick test to verify installation and imports."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lstm_llama.config import get_default_config
from lstm_llama.utils import log_info, get_time_str, get_memory_usage
from lstm_llama.data import FEATURES, load_single_stock_data
from lstm_llama.model import StockLSTM_FinanceLlama
from lstm_llama.evaluate import safe_r2_single, calculate_risk_metrics

import numpy as np


def test_imports():
    config = get_default_config()
    log_info(f"Configuration loaded: input_size={config['input_size']}, "
             f"hidden_size={config['hidden_size']}, output_size={config['output_size']}")
    log_info(f"Number of technical features: {len(FEATURES)}")
    log_info(f"Device: {config['device']}")
    log_info(f"Memory usage: {get_memory_usage():.1f} MB")

    device = config['device']

    model = StockLSTM_FinanceLlama(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        output_size=config['output_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        finance_llama_cache_size=config.get('finance_llama_cache_size', 16),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_info(f"Model created: {total_params:,} total params, "
             f"{trainable_params:,} trainable params")

    batch_size = 4
    seq_length = config['time_steps']
    n_features = config['input_size']
    x_dummy = torch.randn(batch_size, seq_length, n_features).to(device)
    y_dummy = model(x_dummy, config['prompt'])
    log_info(f"Forward pass OK: input {x_dummy.shape} -> output {y_dummy.shape}")

    actual = np.array([100.0, 102.0, 101.0, 103.0, 105.0])
    predicted = np.array([101.0, 101.5, 102.0, 102.5, 104.0])
    r2_val = safe_r2_single(actual, predicted)
    log_info(f"safe_r2_single test: {r2_val:.4f}")

    risk = calculate_risk_metrics(actual.reshape(1, -1), predicted.reshape(1, -1))
    log_info(f"Risk metrics test: volatility={risk['actual_volatility']:.4f}, "
             f"sharpe={risk['actual_sharpe']:.4f}")

    log_info("All tests passed successfully.")
    return True


if __name__ == "__main__":
    import torch
    test_imports()