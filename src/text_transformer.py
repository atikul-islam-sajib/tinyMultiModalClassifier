import os
import sys
import torch
import argparse
import torch.nn as nn
from torchview import draw_graph

sys.path.append("./src/")

from utils import config_files
from transformer_encoder import TransformerEncoder


class TextTransformerEncoder(nn.Module):
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
        super(TextTransformerEncoder, self).__init__()
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
        else:
            self.sequence_length = (self.image_size // self.patch_size) ** 2

        if self.dimension is None:
            self.dimension = (self.patch_size**2) * self.image_size

        self.embedding = nn.Embedding(
            num_embeddings=self.sequence_length, embedding_dim=self.dimension
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
            x = self.embedding(x)
            x = self.transformer_encoder(x)
            return x
        else:
            raise ValueError("Input must be a torch.Tensor.".capitalize())

    @staticmethod
    def total_params(model):
        if isinstance(model, TextTransformerEncoder):
            return sum(params.numel() for params in model.parameters())
        else:
            raise ValueError(
                "Input must be a TextTransformerEncoder model.".capitalize()
            )


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

    sequence_length = (image_size // patch_size) ** 2

    textual_data = torch.randint(0, sequence_length, (1, sequence_length))

    parser = argparse.ArgumentParser(
        description="Text Transformer Encoder for multimodal classification".title()
    )
    parser.add_argument(
        "--nheads",
        type=int,
        default=nheads,
        help="Number of heads for multi-head attention mechanism",
    )
    parser.add_argument(
        "--dimension", type=int, default=dimension, help="Dimension for transformer"
    )
    parser.add_argument(
        "--num_encoder_layers",
        type=int,
        default=num_encoder_layers,
        help="Number of encoder layers",
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=dimension_feedforward,
        help="Dimension for feedforward layers",
    )
    parser.add_argument(
        "--dropout", type=float, default=dropout, help="Dropout probability"
    )
    parser.add_argument(
        "--layer_norm_eps",
        type=float,
        default=layer_norm_eps,
        help="Layer normalization epsilon",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=activation,
        help="Activation function for feedforward layers",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the model for the Text Transformer Encoder".capitalize(),
    )

    args = parser.parse_args()

    text_transfomer = TextTransformerEncoder(
        dimension=args.dimension,
        nheads=args.nheads,
        num_encoder_layers=args.num_encoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        layer_norm_eps=args.layer_norm_eps,
        activation=args.activation,
    )

    assert (text_transfomer(textual_data).size()) == (
        1,
        (image_size // patch_size) ** 2,
        dimension,
    ), "Text Transfomer class is not working properly".capitalize()

    if args.display:
        draw_graph(
            model=text_transfomer,
            input_data=textual_data,
        ).visual_graph.render(
            filename=os.path.join(
                config_files()["artifacts"]["files"], "TextTransformerEncoder"
            ),
            format="pdf",
        )
        print(
            "Text Transformer Encoder architecture saved as TextTransformerEncoder.jpeg in the folder {}".format(
                config_files()["artifacts"]["files"]
            )
        )
        print(
            "Total number of parameters of Text Transformer Encoder {}".format(
                TextTransformerEncoder.total_params(model=text_transfomer)
            )
        )
