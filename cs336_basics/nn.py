import torch
from torch import nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.W = torch.empty(out_features, in_features, dtype=dtype)
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.W, std=std, a=-3 * std, b=3 * std)
        self.W.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T
        