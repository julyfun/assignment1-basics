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
        self.weight = nn.Parameter(torch.ones(d_model)).to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(self.eps + 1 / self.d_model * torch.sum(torch.square(x), dim=-1, keepdim=True))
        return (x / rms * self.weight).to(in_dtype)
        
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
        self.d_ff = d_ff 
        # if d_ff is not None else max(1, int(d_model * 8 / 3 / 64 + 0.5)) * 64
        self.w1 = Linear(d_model, self.d_ff, device, dtype)
        self.w2 = Linear(self.d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, self.d_ff, device, dtype)
       
    def forward(self, x: torch.Tensor) -> Float[Tensor, "... d_model"]:
        return self.w2(silu(self.w1(x)) * (self.w3(x)))
        
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
    
    def forward(self, x: Tensor, token_positions: Tensor | None) -> Tensor:
        """
        x: (..., seq_len, d_k), return same shape
        token_positions:  (..., seq_len) 
        """
        if token_positions is None:
            seq_len = x.shape[-2]
            token_positions = torch.arange(seq_len).expand(*x.shape[:-1])
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
    def __init__(self, d_model: int, num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
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
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        
    def forward(
        self,
        x: Float[Tensor, "b ... seq_len d_model"],
        token_positions: Tensor | None,
        rope: RotaryPositionalEmbedding | None
    ) -> Float[Tensor, "b ... seq_len d_model"]:
        seq_len = x.shape[-2]
        mask = ~torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        # this may be not needed, but it's ok
        mask = mask[(None,) * (x.dim() + 1 - mask.dim()) + (...,)]
        
        h = self.num_heads
        # Wq(x): seq_len, h*dk
        q = rearrange(self.q_proj(x), 'b ... l (h d_k) -> b ... h l d_k', h=h)
        if rope is not None:
            q = rope(q, token_positions) # can be None
        k = rearrange(self.k_proj(x), 'b ... l (h d_k) -> b ... h l d_k', h=h)
        if rope is not None:
            k = rope(k, token_positions)
        v = rearrange(self.v_proj(x), 'b ... l (h d_k) -> b ... h l d_k', h=h)
        multihead = rearrange(
            scaled_dot_product_attention(q, k, v, mask),
            'b ... h l d_k -> b ... l (h d_k)'
        )
        return self.output_proj(multihead)
 
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.attn = MultiheadSelfAttention(
            d_model,
            num_heads,
            device,
            dtype,
        )
        self.ln1 = RMSNorm(
            d_model,
            device=device,
            dtype=dtype,
        )
        self.ffn = SwiGLU(
            d_model,
            d_ff,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(
            d_model,
            device=device,
            dtype=dtype,
        )
        d_k = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(
            theta,
            d_k,
            max_seq_len,
            device=device,
        )
        
    def forward(
        self,
        x: Float[Tensor, "b ... seq_len d_model"],
        token_positions: Tensor | None,
    )-> Float[Tensor, "b ... seq_len d_model"]:
        y = x + self.attn(self.ln1(x), token_positions, rope=self.rope)
        return x + self.ffn(self.ln2(y))
