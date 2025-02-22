import os
import sys
import argparse
import torch
import torch.nn as nn
from torchview import draw_graph

sys.path.append("./src/")

from utils import config_files
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
        activation: str = "relu",
        layer_norm_eps: float = 1e-05,
    ):
        super(TransformerEncoderBlock, self).__init__()
        self.dimension = dimension
        self.nheads = nheads
        self.dim_feedforward = dim_feedforward
        self.activation = activation
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps

        self.multihead_attention = MultiHeadAttentionLayer(
            nheads=self.nheads, dimension=self.dimension
        )
        self.layer_normalization = LayerNormalization(
            dimension=self.dimension, layer_norm_eps=self.layer_norm_eps
        )
        self.feed_forward_network = FeedForwardNeuralNetwork(
            in_features=self.dimension,
            out_features=4 * self.dim_feedforward,
            activation=self.activation,
            dropout=self.dropout,
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

    @staticmethod
    def total_params(model):
        if isinstance(model, TransformerEncoderBlock):
            return sum(params.numel() for params in model.parameters())


if __name__ == "__main__":
    image_channels = config_files()["patchEmbeddings"]["channels"]
    image_size = config_files()["patchEmbeddings"]["image_size"]
    patch_size = config_files()["patchEmbeddings"]["patch_size"]
    dimension = config_files()["patchEmbeddings"]["dimension"]
    nheads = config_files()["transfomerEncoderBlock"]["nheads"]
    activation = config_files()["transfomerEncoderBlock"]["activation"]
    dropout = config_files()["transfomerEncoderBlock"]["dropout"]
    layer_norm_eps = config_files()["transfomerEncoderBlock"]["layer_norm_eps"]
    dim_feedforward = config_files()["transfomerEncoderBlock"]["dimension_feedforward"]
    dropout = config_files()["transfomerEncoderBlock"]["dropout"]

    num_of_patches = (image_size // patch_size) ** 2

    parser = argparse.ArgumentParser(
        description="Transformer Encoder Block for the MultiModal model".title()
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
        help="Please choose from 256, 512, or 1024".capitalize(),
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=1024,
        help="Number of output features for the feed-forward network".capitalize(),
    )
    parser.add_argument(
        "--dropout", type=float, default=dropout, help="Dropout probability"
    )
    parser.add_argument(
        "--activation", type=str, default=activation, help="Activation function"
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the model for the Transformer Encoder Block".capitalize(),
    )

    args = parser.parse_args()

    transformer_encoder_block = TransformerEncoderBlock(
        dimension=dimension,
        nheads=nheads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        layer_norm_eps=layer_norm_eps,
    )

    assert (torch.randn((1, num_of_patches, dimension)).size()) == (
        1,
        num_of_patches,
        dimension,
    ), "Transformer Encoder Block cannot be worked properly".capitalize()

    if args.display:
        draw_graph(
            model=transformer_encoder_block,
            input_data=torch.randn((1, num_of_patches, dimension)),
        ).visual_graph.render(
            filename=os.path.join(
                config_files()["artifacts"]["files"], "TransformerEncoderBlock"
            ),
            format="jpeg",
        )
        print(
            "Transformer Encoder Block diagram saved as TransformerEncoderBlock.jpeg in the folder {}".format(
                config_files()["artifacts"]["files"]
            )
        )
        print(
            "Total parameters for the TransformerEncoderBlock: ",
            TransformerEncoderBlock.total_params(transformer_encoder_block),
        )
