import os
import sys
import argparse
import torch
import torch.nn as nn

sys.path.append("./src/")


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


if __name__ == "__main__":
    network = FeedForwardNeuralNetwork(
        in_features=256,
        out_features=4 * 256,
        dropout=0.1,
        activation="relu",
    )

    print(network(torch.randn((1, 64, 256))).size())
