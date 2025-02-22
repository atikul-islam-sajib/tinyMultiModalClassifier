import os
import sys
import argparse
import torch
from tqdm import tqdm
import torch.nn as nn
from torchview import draw_graph

sys.path.append("./src/")

from utils import config_files
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

    @staticmethod
    def total_params(model):
        if isinstance(model, TransformerEncoder):
            return sum(params.numel() for params in model.parameters())
        else:
            raise ValueError("Input must be a TransformerEncoder model.".capitalize())


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
    layer_norm_eps = config_files()["transfomerEncoderBlock"]["layer_norm_eps"]

    parser = argparse.ArgumentParser(
        description="Transformer Encoder Block for the Multi Model".title()
    )
    parser.add_argument(
        "--nheads",
        type=int,
        default=nheads,
        help="Number of heads for the multi-head attention mechanism".capitalize(),
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=dimension,
        choices=[256, 512, 1024],
    )
    parser.add_argument(
        "--num_encoder_layers",
        type=int,
        default=6,
        help="Number of Transformer Encoder Blocks".capitalize(),
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=1024,
        help="Number of output features for the feed-forward network".capitalize(),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout probability".capitalize(),
    )
    parser.add_argument(
        "--layer_norm_eps",
        type=float,
        default=1e-05,
        help="Layer normalization epsilon".capitalize(),
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        help="Activation function".capitalize(),
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the model for the Transformer Encoder".capitalize(),
    )

    args = parser.parse_args()

    num_of_patches = (image_size // patch_size) ** 2

    transformerEncoder = TransformerEncoder(
        dimension=dimension,
        nheads=nheads,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dimension_feedforward,
        dropout=dropout,
        layer_norm_eps=layer_norm_eps,
        activation=activation,
    )

    assert transformerEncoder(torch.randn((1, num_of_patches, dimension))).size() == (
        1,
        num_of_patches,
        dimension,
    ), "Transformer Encoder Block is not worked due to mistmached of shape, check the arguments please again !!!".capitalize()

    if args.display:
        draw_graph(
            model=transformerEncoder,
            input_data=torch.randn((1, num_of_patches, dimension)),
        ).visual_graph.render(
            filename=os.path.join(
                config_files()["artifacts"]["files"], "TransformerEncoder"
            ),
            format="pdf",
        )
        print(
            "Transformer Encoder diagram saved as TransformerEncoder.jpeg in the folder {}".format(
                config_files()["artifacts"]["files"]
            )
        )
        print(
            "Total number of parameters of Transformer Encoder {}".format(
                TransformerEncoder.total_params(model=transformerEncoder)
            )
        )
