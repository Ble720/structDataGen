import torch
from torch.optim.lr_scheduler import LambdaLR
import math

def get_cosine_warmup_scheduler(optimizer, total_steps, base_lr, warmup_ratio=0.05, min_lr_ratio=0.05):
    warmup_steps = int(total_steps * warmup_ratio)
    min_lr = base_lr * min_lr_ratio

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay with floor
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (min_lr / base_lr) + (1 - min_lr / base_lr) * cosine

    return LambdaLR(optimizer, lr_lambda), total_steps