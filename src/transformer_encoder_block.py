import os
import sys
import warnings
import torch
import torch.nn as nn

sys.path.append("./src/")

from scaled_dot_product import scaled_dot_product


class TransformerEncoderBlock(nn.Module):
    def __init__(self, nheads: int = 8, dimension: int = 256):
        super(TransformerEncoderBlock, self).__init__()
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


if __name__ == "__main__":
    transformerEncoder = TransformerEncoderBlock(
        nheads=8,
        dimension=256,
    )
    attention = transformerEncoder(torch.randn((1, 64, 256)))
    print(attention.size())
