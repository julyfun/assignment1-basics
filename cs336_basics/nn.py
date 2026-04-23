import torch
from torch import nn, Tensor
from jaxtyping import Float, Bool
from collections.abc import Callable
from typing_extensions import override
from einops import rearrange

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype)).to(device)
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, std=std, a=-3 * std, b=3 * std)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T
        
class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.emb = nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=dtype)).to(device)
        nn.init.trunc_normal_(self.emb, std=1, a=-3, b=3)
        
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
        
def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)
    
        
class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        # may use self.d_ff = max(1, int(d_model * 8 / 3 / 64 + 0.5)) * 64
        self.d_ff = d_ff
        self.W1 = Linear(d_model, d_ff, device, dtype)
        self.W2 = Linear(d_ff, d_model, device, dtype)
        self.W3 = Linear(d_model, d_ff, device, dtype)
       
    def forward(self, x: torch.Tensor) -> Float[Tensor, "... d_model"]:
        return self.W2(silu(self.W1(x)) * (self.W3(x)))
        
class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None
    ):
        super().__init__()
        # self.cos = torch.empty(max_seq_len, d_k)
        thetas = torch.tensor(
            [[i / theta ** ((2 * k - 2) / d_k) for k in range(1, d_k // 2 + 1)]
            for i in range(max_seq_len)]
        )
        cos = torch.cos(thetas)
        sin = torch.sin(thetas)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    
    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """
        x: (..., seq_len, d_k), return same shape
        token_positions:  (..., seq_len) 
        """
        ...
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).view_as(x)
 
def softmax(x: Tensor, dim: int) -> Tensor:
    e = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    return e / torch.sum(e, dim=dim, keepdim=True)
    
def scaled_dot_product_attention(
    q: Float[Tensor, "b ... seq_len d_k"],
    k: Float[Tensor, "b ... seq_len d_k"],
    v: Float[Tensor, "b ... seq_len d_v"],
    mask: Bool[Tensor, "b ... seq_len seq_len"] | None,
) -> Float[Tensor, "b ... seq_len d_v"]:
    d_k = q.shape[-1]
    attn = q @ k.mT / d_k ** 0.5 # [b, ..., q_len, k_len]
    if mask is not None:
        mask = mask.expand_as(attn)
        attn[~mask] -= float('inf')
    # v: k_len, d_v
    return softmax(attn, dim=-1) @ v
    
class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # Wq | in : d_model out: h * dk
        #      x: 1 2 3
        # 000    @  head1
        # 000    @
        # ---
        # 000    @  head2
        # 000    @
        self.Wq = Linear(d_model, d_model)
        self.Wk = Linear(d_model, d_model)
        self.Wv = Linear(d_model, d_model)
        self.Wo = Linear(d_model, d_model)
        
    def forward(
        self,
        x: Float[Tensor, "b ... seq_len d_model"],
        token_positions: Tensor | None,
        rope: RotaryPositionalEmbedding | None
    ):
        seq_len = x.shape[-2]
        mask = ~torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask[(None,) * (x.dim() + 1 - mask.dim()) + (...,)]
        
        h = self.num_heads
        # Wq(x): seq_len, h*dk
        # q = self.Wq(x) if (rope is None or token_positions is None) \
        #     else rope(self.Wq(x), token_positions)
        q = rearrange(self.Wq(x), 'b ... l (h d_k) -> b ... h l d_k', h=h)
        if rope is not None and token_positions is not None:
            q = rope(q, token_positions)
        k = rearrange(self.Wk(x), 'b ... l (h d_k) -> b ... h l d_k', h=h)
        if rope is not None and token_positions is not None:
            k = rope(k, token_positions)
        v = rearrange(self.Wv(x), 'b ... l (h d_k) -> b ... h l d_k', h=h)
        multihead = rearrange(
            scaled_dot_product_attention(q, k, v, mask),
            'b ... h l d_k -> b ... l (h d_k)'
        )
        return self.Wo(multihead)
 