import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from .utils import log_info


class StockLSTM_FinanceLlama(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers,
                 dropout=0.1, finance_llama_cache_size=16,
                 llama_model_path='/root/autodl-tmp/Finance-Llama-8B'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.finance_llama_cache = {}
        self.finance_llama_cache_order = []
        self.finance_llama_cache_size = finance_llama_cache_size

        log_info("Loading local Finance-Llama-8B model...")
        import time as _time
        t0 = _time.time()

        config = AutoConfig.from_pretrained(llama_model_path)
        self.finance_llama = AutoModel.from_pretrained(
            llama_model_path, config=config)
        self.finance_llama_tokenizer = AutoTokenizer.from_pretrained(
            llama_model_path)
        if self.finance_llama_tokenizer.pad_token is None:
            self.finance_llama_tokenizer.pad_token = \
                self.finance_llama_tokenizer.eos_token

        for param in self.finance_llama.parameters():
            param.requires_grad = False
        log_info(f"Frozen all Finance-Llama layers (elapsed {_time.time() - t0:.2f}s)")

        if hasattr(self.finance_llama.config, 'hidden_size'):
            model_hidden = self.finance_llama.config.hidden_size
        elif hasattr(self.finance_llama.config, 'n_embd'):
            model_hidden = self.finance_llama.config.n_embd
        else:
            model_hidden = 768
            log_info(f"Could not determine model hidden size, using default: {model_hidden}")
        log_info(f"Model hidden size: {model_hidden}")
        self.finance_llama_fc = nn.Linear(model_hidden, hidden_size // 2)

        self.input_norm = nn.BatchNorm1d(input_size)
        self.input_fc = nn.Linear(input_size, hidden_size // 2)
        lstm_dropout = dropout if num_layers > 1 else 0
        self.lstm1 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=lstm_dropout)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=lstm_dropout)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4,
                                               batch_first=True)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LeakyReLU(),
            nn.Dropout(dropout / 2),
        )
        self.fc3 = nn.Linear(hidden_size // 4, output_size)

        self.init_weights()

        total = sum(p.numel() for p in self.parameters()) / 1_000_000
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad) / 1_000_000
        log_info(f"Total params: {total:.2f}M, Trainable params: {trainable:.2f}M")

    def init_weights(self):
        for module in [self.input_fc, self.finance_llama_fc,
                       self.fc1, self.fc2, self.fc3]:
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.kaiming_normal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0.01)

        for lstm in [self.lstm1, self.lstm2]:
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                    n = param.size(0)
                    param.data[(n // 4):(n // 2)].fill_(1)

    def clear_finance_llama_cache(self):
        self.finance_llama_cache.clear()
        self.finance_llama_cache_order.clear()

    def get_finance_llama_features(self, prompt, device=None):
        if device is None:
            device = next(self.parameters()).device

        if isinstance(prompt, str):
            key = ('STR', prompt)
        elif isinstance(prompt, (list, tuple)):
            key = ('LIST', tuple(prompt))
        else:
            key = ('OTHER', str(prompt))

        if key in self.finance_llama_cache:
            feat = self.finance_llama_cache[key]
            if feat.device != device:
                feat = feat.to(device)
                self.finance_llama_cache[key] = feat
            return feat

        with torch.no_grad():
            texts = [prompt] if isinstance(prompt, str) else list(prompt)
            enc = self.finance_llama_tokenizer(
                texts, return_tensors='pt', padding=True,
                truncation=True, max_length=128)
            input_ids = enc.input_ids.to(device)
            attention_mask = enc.attention_mask.to(device) \
                if 'attention_mask' in enc else None

            if self.finance_llama.device != device:
                self.finance_llama = self.finance_llama.to(device)

            out = self.finance_llama(input_ids=input_ids,
                                     attention_mask=attention_mask)
            last_hidden = out.last_hidden_state[:, -1, :]

            if next(self.finance_llama_fc.parameters()).device != device:
                self.finance_llama_fc = self.finance_llama_fc.to(device)

            feat = self.finance_llama_fc(last_hidden)
            feat = feat.detach()

            self.finance_llama_cache[key] = feat
            self.finance_llama_cache_order.append(key)
            if len(self.finance_llama_cache_order) > self.finance_llama_cache_size:
                oldest = self.finance_llama_cache_order.pop(0)
                self.finance_llama_cache.pop(oldest, None)
            return feat

    def check_device_consistency(self, x, prompt_name="default"):
        device = x.device
        issues = []
        components = {
            'input_norm': self.input_norm, 'input_fc': self.input_fc,
            'finance_llama_fc': self.finance_llama_fc,
            'lstm1': self.lstm1, 'lstm2': self.lstm2,
            'layer_norm1': self.layer_norm1, 'layer_norm2': self.layer_norm2,
            'attention': self.attention, 'fc1': self.fc1,
            'fc2': self.fc2, 'fc3': self.fc3,
        }
        for name, comp in components.items():
            try:
                if next(comp.parameters()).device != device:
                    issues.append(f"{name}: expected {device}, got "
                                  f"{next(comp.parameters()).device}")
            except Exception as e:
                issues.append(f"{name}: error - {e}")
        if hasattr(self, 'finance_llama'):
            try:
                fl_dev = next(self.finance_llama.parameters()).device
                if fl_dev != device:
                    issues.append(f"finance_llama: expected {device}, got {fl_dev}")
            except Exception as e:
                issues.append(f"finance_llama: error - {e}")
        if issues:
            log_info(f"Device issues for {prompt_name}: {'; '.join(issues)}")
            return False
        return True

    def forward(self, x, prompt):
        batch_size = x.size(0)
        seq_length = x.size(1)

        x_reshaped = x.reshape(-1, x.size(-1))
        x_normalized = self.input_norm(x_reshaped)
        x_normalized = x_normalized.reshape(batch_size, seq_length, -1)
        x_features = self.input_fc(x_normalized)

        feat = self.get_finance_llama_features(prompt)
        if feat.size(0) == 1:
            llama_feat = feat.expand(batch_size, -1)
        elif feat.size(0) == batch_size:
            llama_feat = feat
        else:
            p = feat.size(0)
            if p < batch_size:
                reps = (batch_size + p - 1) // p
                llama_feat = feat.repeat(reps, 1)[:batch_size]
            else:
                llama_feat = feat[:batch_size]

        llama_feat = llama_feat.unsqueeze(1).expand(-1, seq_length, -1)
        combined = torch.cat((x_features, llama_feat), dim=2)

        lstm1_out, _ = self.lstm1(combined)
        lstm1_out = self.layer_norm1(lstm1_out)
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = lstm1_out + lstm2_out
        lstm2_out = self.layer_norm2(lstm2_out)

        attn_out, _ = self.attention(lstm2_out, lstm2_out, lstm2_out)
        attn_out = lstm2_out + attn_out
        final_hidden = attn_out[:, -1, :]

        out = self.fc1(final_hidden)
        out = self.fc2(out)
        out = self.fc3(out)
        return out