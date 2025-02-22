import os
import sys
import argparse
import torch
from tqdm import tqdm
import torch.nn as nn

sys.path.append("./src/")

from transformer_encoder_block import TransformerEncoderBlock


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        dimension: int = 256,
        nheads: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-05,
        activation: str = "relu",
    ):
        super(TransformerEncoder, self).__init__()
        self.dimension = dimension
        self.nheads = nheads
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.activation = activation

        self.layers = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    dimension=self.dimension,
                    nheads=self.nheads,
                    dim_feedforward=self.dim_feedforward,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for _ in tqdm(range(self.num_encoder_layers))
            ]
        )

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            for layer in self.layers:
                x = layer(x=x)
            return x
        else:
            raise ValueError("Input must be a torch.Tensor.".capitalize())


if __name__ == "__main__":
    transformerEncoder = TransformerEncoder(
        dimension=256,
        nheads=8,
        num_encoder_layers=8,
        dim_feedforward=1024,
        dropout=0.1,
        layer_norm_eps=1e-05,
        activation="relu",
    )

    print(transformerEncoder(torch.randn((1, 64, 256))).size())
