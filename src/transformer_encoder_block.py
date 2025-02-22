import os
import sys
import warnings
import torch
import torch.nn as nn

sys.path.append("./src/")


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
            x = self.QKV(x)
            return x
        else:
            raise ValueError("Input must be a torch tensor.".capitalize())


if __name__ == "__main__":
    transformerEncoder = TransformerEncoderBlock(
        nheads=8,
        dimension=256,
    )
    print(transformerEncoder(torch.randn((1, 64, 256))).size())
