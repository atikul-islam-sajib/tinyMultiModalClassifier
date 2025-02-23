import os
import sys
import torch
import argparse
import torch.nn as nn

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

    text_transfomer = TextTransformerEncoder(
        dimension=dimension,
        nheads=nheads,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dimension_feedforward,
        dropout=dropout,
        layer_norm_eps=layer_norm_eps,
        activation=activation,
    )

    print(text_transfomer(textual_data).size())
