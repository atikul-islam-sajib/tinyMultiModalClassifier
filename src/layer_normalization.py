import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


class LayerNormalization(nn.Module):
    def __init__(self, dimension: int = 256):
        super(LayerNormalization, self).__init__()

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            pass
        else:
            raise ValueError("Input must be a torch.Tensor.".capitalize())


if __name__ == "__main__":
    pass
