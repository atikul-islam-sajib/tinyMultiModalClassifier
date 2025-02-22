import os
import sys
import warnings
import torch
import argparse
import torch.nn as nn
from torchview import draw_graph

sys.path.append("./src/")

from utils import config_files
from scaled_dot_product import scaled_dot_product


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, nheads: int = 8, dimension: int = 256):
        super(MultiHeadAttentionLayer, self).__init__()
        self.nheads = nheads
        self.dimension = dimension

        warnings.warn(
            "Please ensure that the dimension is a multiple of the number of heads in the encoder block (e.g., 256 % 8 = 0). "
            "This is a requirement for the Transformer Encoder Block to function properly. "
            "If not, you might need to adjust the dimension or the number of heads."
        )
        assert (
            dimension % self.nheads == 0
        ), "Dimension mismatched with nheads and dimension".title()

        self.QKV = nn.Linear(
            in_features=self.dimension, out_features=3 * self.dimension, bias=False
        )

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            QKV = self.QKV(x)
            query, key, values = torch.chunk(input=QKV, chunks=3, dim=-1)
            query = query.view(
                query.size(0), query.size(1), self.nheads, self.dimension // self.nheads
            )
            key = key.view(
                key.size(0), key.size(1), self.nheads, self.dimension // self.nheads
            )
            values = values.view(
                values.size(0),
                values.size(1),
                self.nheads,
                self.dimension // self.nheads,
            )

            query = query.permute(0, 2, 1, 3)
            key = key.permute(0, 2, 1, 3)
            values = values.permute(0, 2, 1, 3)

            attention_output = scaled_dot_product(query=query, key=key, values=values)
            attention_output = attention_output.view(
                attention_output.size(0),
                attention_output.size(2),
                attention_output.size(1) * attention_output.size(-1),
            )

            return attention_output
        else:
            raise ValueError("Input must be a torch tensor.".capitalize())

    @staticmethod
    def total_params(model):
        if isinstance(model, MultiHeadAttentionLayer):
            return sum(params.numel() for params in model.parameters())


if __name__ == "__main__":
    image_channels = config_files()["patchEmbeddings"]["channels"]
    image_size = config_files()["patchEmbeddings"]["image_size"]
    patch_size = config_files()["patchEmbeddings"]["patch_size"]
    dimension = config_files()["patchEmbeddings"]["dimension"]
    nheads = config_files()["transfomerEncoderBlock"]["nheads"]

    parser = argparse.ArgumentParser(
        description="Transformer Encoder Block for the multi modal task".title()
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
        "--display",
        action="store_true",
        default=False,
        help="Display the model of Transformer Block".capitalize(),
    )

    args = parser.parse_args()

    number_of_path_size = (image_size // patch_size) ** 2

    multihead_attention = MultiHeadAttentionLayer(
        nheads=nheads,
        dimension=dimension,
    )
    attention = multihead_attention(torch.randn((1, number_of_path_size, dimension)))
    assert (attention.size()) == (
        1,
        number_of_path_size,
        dimension,
    ), "Transformer encoder failed".capitalize()

    if args.display:
        draw_graph(
            model=multihead_attention,
            input_data=torch.randn((1, number_of_path_size, dimension)),
        ).visual_graph.render(
            filename=os.path.join(
                config_files()["artifacts"]["files"], "MultiHeadAttention"
            ),
            format="pdf",
        )
        print(
            "Transformer encoder block image was saved to ",
            config_files()["artifacts"]["files"],
        )
        print(
            "Total parameters for the TransformerEncoderBlock: ",
            MultiHeadAttentionLayer.total_params(multihead_attention),
        )
