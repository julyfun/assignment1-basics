import argparse
import wandb
from cs336_basics.train import train

parser = argparse.ArgumentParser()
_ = parser.add_argument("--train_data_path", type=str, required=True)
_ = parser.add_argument("--val_data_path", type=str, required=True)
_ = parser.add_argument("--target_iteration", type=int, default=10)
_ = parser.add_argument("--batch_size", type=int, default=64)
_ = parser.add_argument("--context_length", type=int, default=16)
_ = parser.add_argument("--d_model", type=int, default=64)
_ = parser.add_argument("--num_layers", type=int, default=3)
_ = parser.add_argument("--num_heads", type=int, default=4)
_ = parser.add_argument("--d_ff", type=int, default=128)
_ = parser.add_argument("--rope_theta", type=float, default=10000.0)
_ = parser.add_argument("--vocab_size", type=int, required=True)
_ = parser.add_argument("--load_ckpt_path", type=str, default=None)
_ = parser.add_argument("--save_ckpt_path", type=str, default=None)
_ = parser.add_argument("--lr", type=float, default=1e-3)

_ = wandb.init(project="cs336", name="1")

args = parser.parse_args()
train(**vars(args))
