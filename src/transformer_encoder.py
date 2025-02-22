import os
import sys
import argparse
import torch
import torch.nn as nn

sys.path.append("./src/")

from transformer_encoder_block import TransformerEncoderBlock


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        dimension: int = 256,
        nheads: int = 8,
        num_encoder_layers: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super(TransformerEncoder, self).__init__()
        self.dimension = dimension
        self.nheads = nheads
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward

        self.layers = nn.Sequential(
            *[
                TransformerEncoderBlock(nheads=self.nheads, dimension=self.dimension)
                for _ in range(self.num_encoder_layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            pass
        else:
            raise ValueError("Input must be a torch.Tensor.")


if __name__ == "__main__":
    pass
