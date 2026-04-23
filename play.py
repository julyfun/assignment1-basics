import torch
from cs336_basics import nn
from torch import nn as torch_nn


def print_all_submodules(
    module: torch_nn.Module,
    prefix: str = "",
    is_last: bool = True,
    parent_prefix: str = ""
) -> int:
    """
    递归打印模型子模块的树形结构及参数量
    
    Args:
        module: 当前模块
        prefix: 当前行的前缀符号
        is_last: 是否是父模块的最后一个子模块
        parent_prefix: 父模块的前缀（用于缩进）
    
    Returns:
        当前模块的总参数量
    """
    # 计算当前模块的参数量（包括所有子模块）
    total_params = sum(p.numel() for p in module.parameters())
    
    # 构建当前行前缀
    if prefix:
        current_prefix = parent_prefix + prefix
    else:
        current_prefix = parent_prefix
    
    # 获取模块名称
    module_name = module.__class__.__name__
    
    # 打印当前模块
    if total_params > 0:
        params_str = f"{total_params:,}"
        print(f"{current_prefix}{module_name} ({params_str} params)")
    else:
        print(f"{current_prefix}{module_name}")
    
    # 获取命名子模块
    named_children = list(module.named_children())
    
    # 递归打印子模块
    for i, (name, child) in enumerate(named_children):
        is_child_last = (i == len(named_children) - 1)
        
        # 构建子模块的前缀
        if is_child_last:
            child_prefix = "└── "
            child_parent_prefix = parent_prefix + ("    " if prefix else "")
        else:
            child_prefix = "├── "
            child_parent_prefix = parent_prefix + ("│   " if prefix else "")
        
        print_all_submodules(child, child_prefix, is_child_last, child_parent_prefix)
    
    return total_params

    
gpt2_xl = nn.TransformerLM(
    vocab_size=50257,
    context_length=512,
    d_model=1600,
    num_layers=48,
    num_heads=25,
    d_ff=6400,
    rope_theta=10000.0,
    device=torch.device("cpu")
)

print_all_submodules(gpt2_xl)
