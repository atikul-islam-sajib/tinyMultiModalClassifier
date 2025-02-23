import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from utils import config_files


class TextEmbedding(nn.Module):
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
        super(TextEmbedding, self).__init__()
        self.dimension = dimension
        self.nheads = nheads
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.activation = activation

        try:
            self.image_size = config_files()["patchEmbeddings"]["image_size"]
            self.patch_size = config_files()["patchEmbeddings"]["patch_size"]
            self.image_channels = config_files()["patchEmbeddings"]["channels"]
        except KeyError:
            raise ValueError(
                "Image configuration not found in the config files.".capitalize()
            )

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            pass
        else:
            raise ValueError("Input must be a torch.Tensor.".capitalize())


if __name__ == "__main__":
    pass
