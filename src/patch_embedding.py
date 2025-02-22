import os
import sys
import torch
import torch.nn as nn

sys.path.append("/src/")


class PatchEmbedding:
    def __init__(
        self,
        channels: int = 3,
        patch_size: int = 16,
        image_size: int = 128,
        dimension: int = None,
    ):
        self.in_channels = channels
        self.patch_size = patch_size
        self.dimension = dimension
        self.image_size = image_size

        if self.dimension is None:
            self.dimension = (self.patch_size**2) * self.in_channels

        self.number_of_pathches = (self.image_size // self.patch_size) ** 2

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            pass
        else:
            raise ValueError("Input must be a torch tensor.".capitalize())


if __name__ == "__main__":
    pass
