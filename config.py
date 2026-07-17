import os
import torch


def get_default_config():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return {
        'input_size': 19,
        'hidden_size': 128,
        'output_size': 5,
        'num_layers': 2,
        'dropout': 0.15,
        'learning_rate': 0.0001,
        'batch_size': 512,
        'epochs': 10,
        'time_steps': 30,
        'prompt': (
            "You are a financial expert predicting stock prices. "
            "Consider these factors:\n"
            "- Historical price trends and patterns\n"
            "- Technical indicators (MA, RSI, MACD, Bollinger Bands)\n"
            "- Market volatility and volume changes\n"
            "- Support and resistance levels\n"
            "- Market sentiment and momentum indicators"
        ),
        'device': device,
        'output_dir': './test_output',
        'model_path': './test_output/best_model_r2.pth',
        'data_dir': './data',
        'llama_model_path': '/root/autodl-tmp/Finance-Llama-8B',
    }


def ensure_dirs(config):
    os.makedirs(config['output_dir'], exist_ok=True)