import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from utils import config_files
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
        dropout: float = 0.1,
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

        self.vision_transformer = VisionTransformer(
            channels=channels,
            patch_size=patch_size,
            image_size=image_size,
            dimension=dimension,
            nheads=nheads,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            activation=activation,
        )

        self.text_transformer = TextTransformerEncoder(
            dimension=self.dimension,
            nheads=self.nheads,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            layer_norm_eps=self.layer_norm_eps,
            activation=self.activation,
        )

    def forward(self, image: torch.Tensor, text: torch.Tensor):
        if isinstance(image, torch.Tensor) and isinstance(text, torch.Tensor):
            image_features = self.vision_transformer(image)
            text_features = self.text_transformer(text)

            return {"image_features": image_features, "text_features": text_features}
        else:
            raise ValueError("Both inputs must be torch.Tensor.".capitalize())


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

    number_of_patches = (image_size // patch_size) ** 2
    number_of_sequences = (image_size // patch_size) ** 2

    images = torch.randn(1, image_channels, image_size, image_size)
    texts = torch.randint(0, number_of_sequences, (1, number_of_sequences))

    classifier = MultiModalClassifier(
        channels=image_channels,
        patch_size=patch_size,
        image_size=image_size,
        dimension=dimension,
        nheads=nheads,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dimension_feedforward,
        dropout=dropout,
        layer_norm_eps=layer_norm_eps,
        activation=activation,
    )
    output = classifier(image=images, text=texts)
    print(output["image_features"].size(), output["text_features"].size())
