import os
import sys
import argparse
import torch
import torch.nn as nn

sys.path.append("./src/")

from multihead_attention_layer import MultiHeadAttentionLayer
from feed_forward_network import FeedForwardNeuralNetwork
from layer_normalization import LayerNormalization


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        dimension: int = 256,
        nheads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super(TransformerEncoderBlock, self).__init__()
        self.dimension = dimension
        self.nheads = nheads
        self.dim_feedforward = dim_feedforward

        self.multihead_attention = MultiHeadAttentionLayer(
            nheads=self.nheads, dimension=self.dimension
        )
        self.layer_normalization = LayerNormalization(dimension=self.dimension)
        self.feed_forward_network = FeedForwardNeuralNetwork(
            in_features=self.dimension, out_features=4 * self.dim_feedforward
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            residual = x

            x = self.multihead_attention(x)
            x = self.dropout(x)
            x = torch.add(x, residual)
            x = self.layer_normalization(x)

            residual = x

            x = self.feed_forward_network(x)
            x = self.dropout(x)
            x = torch.add(x, residual)
            x = self.layer_normalization(x)

            return x
        else:
            raise ValueError("Input must be a torch.Tensor.")


if __name__ == "__main__":
    transformer_encoder_block = TransformerEncoderBlock(
        dimension=256,
        nheads=8,
    )
    print(torch.randn((1, 64, 256)).size())
