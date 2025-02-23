import os
import sys
import torch
import argparse
import torch.nn as nn
from torchview import draw_graph

sys.path.append("./src/")

from utils import config_files
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

    @staticmethod
    def total_params(model):
        if isinstance(model, VisionTransformer):
            return sum(params.numel() for params in model.parameters())
        else:
            raise ValueError("Input must be a VisionTransformer model.".capitalize())


if __name__ == "__main__":
    image_channels = config_files()["patchEmbeddings"]["channels"]
    image_size = config_files()["patchEmbeddings"]["image_size"]
    patch_size = config_files()["patchEmbeddings"]["patch_size"]
    dimension = config_files()["patchEmbeddings"]["dimension"]
    nheads = config_files()["transfomerEncoderBlock"]["nheads"]
    activation = config_files()["transfomerEncoderBlock"]["activation"]
    dropout = config_files()["transfomerEncoderBlock"]["dropout"]
    num_encoder_layers = config_files()["transfomerEncoderBlock"]["num_encoder_layers"]
    dimension_feedforward = config_files()["transfomerEncoderBlock"][
        "dimension_feedforward"
    ]
    layer_norm_eps = float(config_files()["transfomerEncoderBlock"]["layer_norm_eps"])

    parser = argparse.ArgumentParser(
        description="Vision Transformer for multimodal model".title()
    )

    parser.add_argument(
        "--image_channels",
        type=int,
        default=image_channels,
        help="Image channels to transform".capitalize(),
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=image_size,
        help="Image size to transform".capitalize(),
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=patch_size,
        help="Patch size for the transformer".capitalize(),
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=dimension,
        help="Dimension for the transformer".capitalize(),
    )
    parser.add_argument(
        "--nheads",
        type=int,
        default=nheads,
        help="Number of heads for the multi-head attention".capitalize(),
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=activation,
        help="Activation function for the transformer".capitalize(),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=dropout,
        help="Dropout probability for the transformer".capitalize(),
    )
    parser.add_argument(
        "--num_encoder_layers",
        type=int,
        default=num_encoder_layers,
        help="Number of layers in the transformer".capitalize(),
    )
    parser.add_argument(
        "--dimension_feedforward",
        type=int,
        default=dimension_feedforward,
        help="Dimension for the feedforward layer".capitalize(),
    )
    parser.add_argument(
        "--layer_norm_eps",
        type=float,
        default=layer_norm_eps,
        help="Layer normalization epsilon".capitalize(),
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the model architectitecture of ViT".capitalize(),
    )

    args = parser.parse_args()

    image = torch.randn((1, image_channels, image_size, image_size))
    num_of_patches = (image_size // patch_size) ** 2

    vision_transformer = VisionTransformer(
        channels=args.image_channels,
        patch_size=args.patch_size,
        image_size=args.image_size,
        dimension=args.dimension,
        nheads=args.nheads,
        num_encoder_layers=args.num_encoder_layers,
        dim_feedforward=args.dimension_feedforward,
        dropout=args.dropout,
        layer_norm_eps=args.layer_norm_eps,
        activation=args.activation,
    )
    assert (vision_transformer(image).size()) == (
        1,
        num_of_patches,
        dimension,
    ), "ViT is not working properly".capitalize()

    if args.display:
        draw_graph(
            model=vision_transformer,
            input_data=image,
        ).visual_graph.render(
            filename=os.path.join(config_files()["artifacts"]["files"], "ViT"),
            format="pdf",
        )
        print(
            "ViT diagram saved to ",
            config_files()["artifacts"]["files"],
        )
        print(
            "Total number of parameters: ",
            VisionTransformer.total_params(vision_transformer),
        )
