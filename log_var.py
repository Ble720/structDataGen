import torch
import torch.nn as nn

class UncertaintyWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_var_cat = nn.Parameter(torch.tensor([0.5]))
        self.log_var_bin = nn.Parameter(torch.tensor([0.5]))
        self.log_var_off = nn.Parameter(torch.tensor([0.0]))