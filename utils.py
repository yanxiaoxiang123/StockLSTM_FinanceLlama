import os
import time
import datetime
import psutil
import torch


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_time_str():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def log_info(message):
    print(f"[{get_time_str()}] {message}")


def load_model_weights(model, weight_path, device):
    model.load_state_dict(torch.load(weight_path, map_location=device))
    return model