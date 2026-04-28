from collections.abc import Callable, Iterable
from typing import Optional, TypeAlias, Union, Dict, Any, Tuple
from torch import nn, Tensor
import torch
import math

ParamsT: TypeAlias = Union[
    Iterable[torch.Tensor], Iterable[Dict[str, Any]], Iterable[Tuple[str, torch.Tensor]]
]

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr} # can be accessed as group["lr"]
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss
        

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data # Get the gradient of loss with respect to p.
                
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                m = state.get("m", torch.zeros_like(grad))
                v = state.get("v", torch.zeros_like(grad))
                
                t += 1
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2
                
                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data -= lr_t * m / (torch.sqrt(v) + eps) # Update weight tensor in-place.
                p.data -= lr * weight_decay * p.data
                
                state["t"] = t # Increment iteration number.
                state["m"] = m
                state["v"] = v
                
        return loss
 
def lr_cosine_schedule(
    t, a_max, a_min, t_w, t_c
):
    if t < t_w:
        return t / t_w * a_max
    if t > t_c:
        return a_min
    return a_min + 0.5 * (1.0 + math.cos(math.pi * (t - t_w) / (t_c - t_w))) * (a_max - a_min)
    

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    eps = 1e-6
    grads = []
    for param in parameters:
        if param.grad is None:
            continue
        grads.append(param.grad.data)
        
    if not grads:
        return
    
    norm = torch.norm(
        torch.cat([g.view(-1) for g in grads])
    )
    
    if norm > max_l2_norm:
        for param in parameters:
            if param.grad is None:
                continue
            param.grad.data *= max_l2_norm / (norm + eps)
 