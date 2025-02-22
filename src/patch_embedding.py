import os
import sys
import torch
import argparse
import torch.nn as nn
from torchview import draw_graph

sys.path.append("/src/")

from utils import config_files


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

    @staticmethod
    def total_params(model):
        if isinstance(model, PatchEmbedding):
            return sum(params.numel() for params in model.parameters())


if __name__ == "__main__":
    image_channels = config_files()["patchEmbeddings"]["channels"]
    image_size = config_files()["patchEmbeddings"]["image_size"]
    patch_size = config_files()["patchEmbeddings"]["patch_size"]
    dimension = config_files()["patchEmbeddings"]["dimension"]

    parser = argparse.ArgumentParser(
        description="Patch Embedding for the ViT - MultiModal Model".title()
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=image_channels,
        help="Number of channels for the Image".capitalize(),
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=patch_size,
        help="Size of the patches".capitalize(),
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=image_size,
        help="Size of the Image".capitalize(),
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=dimension,
        help="Dimensionality of the embeddings".capitalize(),
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="To display the Patch Embedding for the ViT - MultiModal Model".capitalize(),
    )

    args = parser.parse_args()

    image = torch.randn(
        (image_channels // image_channels, image_channels, image_size, image_size)
    )
    patchEmbedding = PatchEmbedding(
        channels=image_channels,
        patch_size=patch_size,
        image_size=image_size,
        dimension=dimension,
    )
    assert patchEmbedding(image).size() == (
        image_channels // image_channels,
        (image_size // patch_size) ** 2,
        dimension,
    ), "PatchEmbedding dimension mismatch".capitalize()

    if args.display:
        print(
            "Total parameters in PatchEmbedding: {}".format(
                PatchEmbedding.total_params(patchEmbedding)
            )
        )

        draw_graph(
            model=patchEmbedding,
            input_data=torch.randn(
                (
                    image_channels // image_channels,
                    image_channels,
                    image_size,
                    image_size,
                )
            ),
        ).visual_graph.render(
            filename=os.path.join(
                config_files()["artifacts"]["files"], "PathEmbedding"
            ),
            format="pdf",
        )

        print(
            "Patch Embedding diagram has been saved in {}".format(
                os.path.join(config_files()["artifacts"]["files"], "PathEmbedding.pdf")
            )
        )
