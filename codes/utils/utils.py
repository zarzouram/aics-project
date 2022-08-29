from typing import List

import os
import random
import json
import numpy as np
import torch


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config_file(config_path: str) -> List[dict]:
    with open(config_path, "r") as f:
        configs = json.load(f)

    return configs


def save_config_file(config_path: str, data: dict) -> List[dict]:
    with open(config_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)