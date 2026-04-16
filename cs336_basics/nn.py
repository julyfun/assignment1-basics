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
        
class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        W = torch.empty(num_embeddings, embedding_dim, dtype=dtype)
        nn.init.trunc_normal_(W, std=1, a=-3, b=3)
        self.emb = nn.Parameter(W)
        self.emb.to(device)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.emb[token_ids, :]
        
class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d_model)).to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(self.eps + 1 / self.d_model * torch.sum(torch.square(x), dim=-1, keepdim=True))
        return (x / rms * self.g).to(in_dtype)
        