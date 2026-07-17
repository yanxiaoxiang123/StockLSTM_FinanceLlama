
## 1. Overview

This repository contains the implementation of the **Qwen-LSTM** hybrid model

## 2. Repository Structure

```
├── config.py       # Hyperparameters and device configuration
├── utils.py        # Logging, memory monitoring, weight loading
├── data.py         # Data loading, feature engineering, preprocessing, splitting
├── model.py        # Qwen-LSTM hybrid model definition
├── train.py        # Training loop, scheduler, early stopping
├── evaluate.py     # Evaluation metrics and risk metrics
├── backtest.py     # Backtesting strategy and visualization
├── predict.py      # Future prediction module
├── main.py         # Pipeline orchestration
├── test_imports.py # Quick test script
└── README.md       # This file
```

## 3. Installation

### 3.1 Prerequisites

- Python 3.10+
- CUDA-capable GPU with 24GB+ VRAM (recommended)
- Conda or pip

### 3.2 Setup

```bash
git clone https://github.com/yanxiaoxiang123/StockLSTM_FinanceLlama.git
cd StockLSTM_FinanceLlama
pip install torch transformers scikit-learn pandas numpy matplotlib tqdm psutil
```

### 3.3 Model Weights

Download Finance-Llama-8B and set the path in `config.py`:

```python
'llama_model_path': '/path/to/your/llm'
```

## 4. Usage

### 4.1 Data Acquisition

Stock data can be obtained via:

- **Tushare**: `pip install tushare` — Chinese stock market API. Register at https://tushare.pro to get a token. Example: `pro = ts.pro_api('your_token'); df = pro.daily(ts_code='000001.SZ', start_date='20200101', end_date='20241231')`
- **AKShare**: `pip install akshare` — open-source Chinese financial data interface. Example: `df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20200101", end_date="20241231", adjust="qfq")`

### 4.2 Data Preparation

Prepare CSV files with columns: `trade_date`, `open`, `high`, `low`, `close`, `vol`. Place them in `./data/`.

### 4.3 Run Full Pipeline

```bash
python -m lstm_llama.main
```

This executes:
1. Load and process data
2. Compute technical indicators
3. Create sliding window sequences
4. Global timestamp-based train/val/test split
5. Fit scalers on training set only
6. Initialize and train the Qwen-LSTM model
7. Evaluate with full metrics
8. Run backtesting simulation

### 4.4 Quick Test

```bash
python test_imports.py
```

Expected output:
```
[timestamp] Configuration loaded: input_size=19, hidden_size=128, output_size=5
[timestamp] Model created: X,XXX,XXX total params, X,XXX,XXX trainable params
[timestamp] Forward pass OK: input torch.Size([4, 30, 19]) -> output torch.Size([4, 5])
[timestamp] All tests passed successfully.
```


