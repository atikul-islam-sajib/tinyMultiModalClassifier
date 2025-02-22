import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


class LayerNormalization(nn.Module):
    def __init__(self, dimension: int = 256):
        super(LayerNormalization, self).__init__()
        self.dimension = dimension

        self.alpha = nn.Parameter(
            data=torch.randn((1, 1, self.dimension)), requires_grad=True
        )
        self.beta = nn.Parameter(
            data=torch.randn((1, 1, self.dimension)), requires_grad=True
        )

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            mean = torch.mean(x, dim=-1).unsqueeze(-1)
            variance = torch.var(x, dim=-1).unsqueeze(-1)

            return self.alpha * mean + self.beta * variance

        else:
            raise ValueError("Input must be a torch.Tensor.".capitalize())


if __name__ == "__main__":
    layer_normalization = LayerNormalization(dimension=256)
    print(layer_normalization(torch.randn((1, 64, 256))).size())
