import numpy.typing as npt
import torch
from torch import Tensor
import random
import os
import typing

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(dataset) - context_length - 1 # datasetlen = 6, len = 5 => only n = 0 is accepted
    xs: list[Tensor] = []
    ys : list[Tensor] = []
    for _ in range(batch_size):
        st = random.randint(0, n)
        xs.append(torch.tensor(dataset[st:st + context_length], dtype=torch.long))
        ys.append(torch.tensor(dataset[st + 1:st + context_length + 1], dtype=torch.long))
    return torch.stack(xs).to(device), torch.stack(ys).to(device)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    d = torch.load(src)
    model.load_state_dict(d["model"])
    optimizer.load_state_dict(d["optimizer"])
    return d["iteration"]
