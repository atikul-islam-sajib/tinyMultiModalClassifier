import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from patch_embedding import PatchEmbedding
from transformer_encoder import TransformerEncoder


class VisionTransformer(nn.Module):
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
        super(VisionTransformer, self).__init__()

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

        self.patch_embedding = PatchEmbedding(
            channels=channels,
            patch_size=patch_size,
            image_size=image_size,
            dimension=dimension,
        )
        self.transformer_encoder = TransformerEncoder(
            dimension=self.dimension,
            nheads=self.nheads,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            layer_norm_eps=self.layer_norm_eps,
            activation=self.activation,
        )

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            x = self.patch_embedding(x)
            x = self.transformer_encoder(x)
            return x
        else:
            raise ValueError("Input must be a torch.Tensor.".capitalize())


if __name__ == "__main__":
    image = torch.randn((1, 3, 128, 128))
    vision_transformer = VisionTransformer(
        channels=3,
        patch_size=16,
        image_size=128,
        dimension=256,
        nheads=8,
        num_encoder_layers=6,
        dim_feedforward=1024,
        dropout=0.0,
        layer_norm_eps=1e-05,
        activation="relu",
    )
    print(vision_transformer(image).size())
