import os
import sys
import argparse
import torch
import torch.nn as nn
from torchview import draw_graph

sys.path.append("./src/")

from utils import config_files


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(
        self,
        in_features: int = 256,
        out_features: int = 4 * 256,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super(FeedForwardNeuralNetwork, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.activation = activation

        if activation == "relu":
            self.activation_function = nn.ReLU()
        elif activation == "gelu":
            self.activation_function = nn.GELU()
        elif activation == "selu":
            self.activation_function = nn.SELU(inplace=True)
        else:
            raise ValueError(
                "Invalid activation function. Choose from'relu', 'gelu', or'selu'."
            )

        self.layers = []

        for idx in range(2):
            self.layers.append(
                nn.Linear(in_features=self.in_features, out_features=self.out_features)
            )
            self.in_features = self.out_features
            self.out_features = in_features

            if idx == 0:
                self.layers.append(self.activation_function)
                self.layers.append(nn.Dropout(self.dropout))

        self.network = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            return self.network(x)
        else:
            raise ValueError("Input must be a torch.Tensor.")

    @staticmethod
    def total_params(model):
        if isinstance(model, FeedForwardNeuralNetwork):
            return sum(params.numel() for params in model.parameters())


if __name__ == "__main__":
    image_channels = config_files()["patchEmbeddings"]["channels"]
    image_size = config_files()["patchEmbeddings"]["image_size"]
    patch_size = config_files()["patchEmbeddings"]["patch_size"]
    dimension = config_files()["patchEmbeddings"]["dimension"]
    nheads = config_files()["transfomerEncoderBlock"]["nheads"]
    dropout = config_files()["transfomerEncoderBlock"]["dropout"]
    activation = config_files()["transfomerEncoderBlock"]["activation"]

    number_of_patches = (image_size // patch_size) ** 2

    parser = argparse.ArgumentParser(description="Feedforward Neural Network".title())
    parser.add_argument(
        "--in_features", type=int, default=dimension, help="Number of input features"
    )
    parser.add_argument(
        "--out_features",
        type=int,
        default=4 * dimension,
        help="Number of output features",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout probability"
    )
    parser.add_argument(
        "--activation", type=str, default=activation, help="Activation function"
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the model architecture of FFNN".capitalize(),
    )

    args = parser.parse_args()

    network = FeedForwardNeuralNetwork(
        in_features=dimension,
        out_features=4 * dimension,
        dropout=dropout,
        activation=activation,
    )

    assert (network(torch.randn((1, number_of_patches, dimension))).size()) == (
        1,
        number_of_patches,
        dimension,
    ), "FFNN network must have exactly = dimension".capitalize()

    if args.display:
        draw_graph(
            model=network, input_data=torch.randn((1, number_of_patches, dimension))
        ).visual_graph.render(
            filename=os.path.join(config_files()["artifacts"]["files"], "FFNN"),
            format="pdf",
        )
        print(
            "Feedforward Neural Network architecture saved as FFNN.jpeg in the folder {}".format(
                config_files()["artifacts"]["files"]
            )
        )
        print(
            "Total number of parameters of FFNN {}".format(
                FeedForwardNeuralNetwork.total_params(model=network)
            )
        )
