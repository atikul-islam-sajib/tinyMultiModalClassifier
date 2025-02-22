import os
import sys
import math
import torch
import torch.nn as nn

sys.path.append("./src/")


def scaled_dot_product(query: torch.Tensor, key: torch.Tensor, values: torch.Tensor):
    if (
        isinstance(query, torch.Tensor)
        and isinstance(key, torch.Tensor)
        and isinstance(values, torch.Tensor)
    ):
        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.size(-1))
        attention_weights = torch.softmax(logits, dim=-1)
        attention_output = torch.matmul(attention_weights, values)
        return attention_output

    else:
        raise ValueError("All inputs must be torch tensors.".capitalize())


if __name__ == "__main__":
    attention = scaled_dot_product(
        torch.randn((1, 8, 64, 32)),
        torch.randn((1, 8, 64, 32)),
        torch.randn((1, 8, 64, 32)),
    )
    print(attention.size())  # (1, 8, 64, 32)
