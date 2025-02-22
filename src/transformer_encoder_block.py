import os
import sys
import torch
import torch.nn as nn

sys.path.append("./src/")


class TransformerEncoderBlock(nn.Module):
    def __init__(self, nheads: int = 8, dimension: int = 256):
        super(TransformerEncoderBlock, self).__init__()
        self.nheads = nheads
        self.dimension = dimension

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            pass
        else:
            raise ValueError("Input must be a torch tensor.".capitalize())


if __name__ == "__main__":
    pass
