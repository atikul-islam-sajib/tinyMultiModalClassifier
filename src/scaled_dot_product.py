import os
import sys
import torch
import torch.nn as nn

sys.path.append("./src/")


def scaled_dot_product(query: torch.Tensor, key: torch.Tensor, values: torch.Tensor):
    if (
        isinstance(query, torch.Tensor)
        and isinstance(key, torch.Tensor)
        and isinstance(values, torch.Tensor)
    ):
        pass
    else:
        raise ValueError("All inputs must be torch tensors.".capitalize())
