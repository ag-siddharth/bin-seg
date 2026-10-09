import yaml
import os
import copy
import torch
import random
import numpy as np
from datetime import datetime  # used in some logging, keeping for now


def load_config(config_path):
    """Load a YAML config, resolving _base_ inheritance."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    if '_base_' in cfg:
        base_path = cfg.pop('_base_')
        # resolve relative to the config file's directory
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(config_path), base_path)
        base_cfg = load_config(base_path)
        cfg = _merge_dicts(base_cfg, cfg)

    return cfg


def _merge_dicts(base, override):
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge_dicts(result[k], v)
        else:
            result[k] = v
    return result


def save_checkpoint(model, optimizer, epoch, metrics, path):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    torch.save(state, path)
    # print(f"[ckpt] saved to {path}")  # noisy, turn on if debugging saves


def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # TODO: torch.backends.cudnn.deterministic? slows things down a lot


def collate_fn(batch):
    # can't use default collate because targets have variable number of objects
    return tuple(zip(*batch))


class AverageMeter:
    """Running average tracker. Nothing fancy."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_device():
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        print("Using GPU: %s" % torch.cuda.get_device_name(0))
    else:
        dev = torch.device('cpu')
        print("CUDA not available, using CPU (this will be slow)")
    return dev
