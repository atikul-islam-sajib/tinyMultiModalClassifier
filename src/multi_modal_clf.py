import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from ViT import VisionTransformer
from text_transformer import TextTransformerEncoder


class MultiModalClassifier(nn.Module):
    def __init__(
        self,
        channels: int = 3,
        patch_size: int = 16,
        image_size: int = 128,
        dimension: int = 256,
        nheads: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-05,
        activation: str = "relu",
    ):
        super(MultiModalClassifier, self).__init__()

        self.channels = channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.dimension = dimension
        self.nheads = nheads
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.activation = activation

    def forward(self, image: torch.Tensor, text: torch.Tensor):
        if isinstance(image, torch.Tensor) and isinstance(text, torch.Tensor):
            pass
        else:
            raise ValueError("Both inputs must be torch.Tensor.".capitalize())


if __name__ == "__main__":
    pass
