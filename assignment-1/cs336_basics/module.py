import torch 
import torch.nn as nn

class Linear(nn.Module): 
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None): 
        super().__init__()
        meta = {'device': device, 'dtype': dtype}

        weight_tensor = torch.empty((in_features, out_features), **meta)
        nn.init.trunc_normal_(weight_tensor)

        self.W = nn.Parameter(weight_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return torch.matmul(x, self.W)