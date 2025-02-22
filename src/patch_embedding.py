import os
import sys
import torch
import torch.nn as nn
from torchview import draw_graph

sys.path.append("/src/")


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        channels: int = 3,
        patch_size: int = 16,
        image_size: int = 128,
        dimension: int = None,
    ):
        super(PatchEmbedding, self).__init__()

        self.in_channels = channels
        self.patch_size = patch_size
        self.dimension = dimension
        self.image_size = image_size

        if self.dimension is None:
            self.dimension = (self.patch_size**2) * self.in_channels

        self.number_of_pathches = (self.image_size // self.patch_size) ** 2

        self.kernel_size, self.stride_size = self.patch_size, self.patch_size
        self.padding_size = self.patch_size // self.patch_size

        self.projection = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.dimension,
            kernel_size=self.kernel_size,
            stride=self.stride_size,
            padding=self.padding_size,
            bias=False,
        )

        self.positional_embeddings = nn.Parameter(
            torch.randn(self.padding_size, self.number_of_pathches, self.dimension),
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            x = self.projection(x)
            x = x.view(x.size(0), x.size(1), -1)
            x = x.permute(0, 2, 1)
            x = self.positional_embeddings + x
            return x
        else:
            raise ValueError("Input must be a torch tensor.".capitalize())


if __name__ == "__main__":
    image = torch.randn((1, 3, 128, 128))
    patchEmbedding = PatchEmbedding(
        channels=3, patch_size=16, image_size=128, dimension=256
    )
    print(patchEmbedding(image).size())
