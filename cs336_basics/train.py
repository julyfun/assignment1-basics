import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
import random
import os
from tqdm import tqdm
import wandb
from typing import Union
from cs336_basics import nn, data, optim

def train(
    *,
    train_data_path: str,
    val_data_path: str,
    target_iteration: int,
    # train
    batch_size: int,
    # model
    d_model: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    vocab_size: int,
    context_length: int,
    num_layers: int,
    # save
    load_ckpt_path: str | None = None,
    save_ckpt_path: str | None = None,
    lr: Union[float, Tensor] = 1e-3,
):
    train_dataset = np.load(train_data_path, mmap_mode="r")
    val_dataset = np.load(val_data_path, mmap_mode="r")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def get_model():
        model = nn.TransformerLM(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
            vocab_size=vocab_size,
            context_length=context_length,
            num_layers=num_layers,
        )
        opt = optim.AdamW(model.parameters(), lr=lr)
        iteration = 0
        if load_ckpt_path:
            iteration = data.load_checkpoint(load_ckpt_path, model, opt)
        return model, opt, iteration
        
    model, opt, iteration = get_model()
    log_steps = 0
    pbar = tqdm(range(iteration, target_iteration))
    for epoch in pbar:
        model.train()
        log_train_losses = []
        for x, y in data.get_batch(
            train_dataset,
            batch_size,
            context_length,
            device,
        ):
            opt.zero_grad()
            logits = model(x)
            loss = nn.cross_entropy_loss(logits, y)
            loss.backward()
            opt.step()
            
            log_steps += 1
            log_train_losses.append(loss.item()) # item(): convert the tensor to a scalar (on cpu)
            
        model.eval()
        log_eval_losses = []
        for x, y in data.get_batch(
            val_dataset,
            batch_size,
            context_length,
            device,
        ):
            opt.zero_grad()
            logits = model(x)
            loss = nn.cross_entropy_loss(logits, y)
            loss.backward()
            opt.step()
            
            log_eval_losses.append(loss.item())
            
        log_trn = sum(log_train_losses) / len(log_train_losses)
        log_val = sum(log_eval_losses) / len(log_eval_losses)
        pbar.set_postfix({"train": log_trn, "val": log_val})
        wandb.log(
            {
                "train/loss": log_trn,
                "val/loss": log_val,
                "epoch": epoch,
                "steps": log_steps,
            }
        )
        
    if save_ckpt_path is not None:
        data.save_checkpoint(model, opt, iteration, save_ckpt_path)
