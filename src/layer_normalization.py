import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from utils import config_files


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
    image_channels = config_files()["patchEmbeddings"]["channels"]
    image_size = config_files()["patchEmbeddings"]["image_size"]
    patch_size = config_files()["patchEmbeddings"]["patch_size"]
    dimension = config_files()["patchEmbeddings"]["dimension"]
    nheads = config_files()["transfomerEncoderBlock"]["nheads"]
    activation = config_files()["transfomerEncoderBlock"]["activation"]
    dropout = config_files()["transfomerEncoderBlock"]["dropout"]

    num_of_patches = (image_size // patch_size) ** 2

    parser = argparse.ArgumentParser(
        description="Layer configuration for the Transfomer Encoder".title()
    )
    parser.add_argument(
        "--dimension", type=int, default=dimension, help="Number of output features"
    )

    args = parser.parse_args()

    layer_normalization = LayerNormalization(dimension=dimension)

    assert (layer_normalization(torch.randn((1, patch_size, dimension))).size()) == (
        1,
        patch_size,
        dimension,
    ), "Invalid layer normalization".capitalize()
