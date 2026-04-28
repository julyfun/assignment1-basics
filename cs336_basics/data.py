import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
import random

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(dataset) - context_length - 1 # datasetlen = 6, len = 5 => only n = 0 is accepted
    xs: list[Tensor] = []
    ys : list[Tensor] = []
    for _ in range(batch_size):
        st = random.randint(0, n)
        xs.append(torch.tensor(dataset[st:st + context_length]))
        ys.append(torch.tensor(dataset[st + 1:st + context_length + 1]))
    return torch.stack(xs).to(device), torch.stack(ys).to(device)
 