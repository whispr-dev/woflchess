import os
from pathlib import Path

def ensure_dir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p

def device_select():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
