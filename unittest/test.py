import os
import sys
import torch
import unittest
import torch.nn as nn

sys.path.append("./src/")

from utils import config_files
from patch_embedding import PatchEmbedding
from scaled_dot_product import scaled_dot_product
from multihead_attention_layer import MultiHeadAttentionLayer


class UnitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.image_channels = config_files()["patchEmbeddings"]["channels"]
        self.image_size = config_files()["patchEmbeddings"]["image_size"]
        self.patch_size = config_files()["patchEmbeddings"]["patch_size"]
        self.dimension = config_files()["patchEmbeddings"]["dimension"]
        self.nheads = config_files()["transfomerEncoderBlock"]["nheads"]

        self.number_of_patch_size = (self.image_size // self.patch_size) ** 2

        self.Q, self.K, self.V = (
            torch.randn(
                1,
                self.nheads,
                self.number_of_patch_size,
                self.dimension // self.nheads,
            )
            for _ in range(3)
        )
        self.pathEmbedding = PatchEmbedding(
            channels=self.image_channels,
            patch_size=self.patch_size,
            image_size=self.image_size,
            dimension=self.dimension,
        )

        self.pathEmbedding1 = PatchEmbedding(
            channels=self.image_channels,
            patch_size=self.patch_size,
            image_size=self.image_size,
        )
        self.attention = scaled_dot_product(
            query=self.Q,
            key=self.K,
            values=self.V,
        )
        self.multihead_attention = MultiHeadAttentionLayer(
            nheads=self.nheads,
            dimension=self.dimension,
        )

    def test_pathEmebeddingLayer(self):
        self.assertEqual(
            self.pathEmbedding(
                torch.randn(
                    (
                        self.image_channels // self.image_channels,
                        self.image_channels,
                        self.image_size,
                        self.image_size,
                    )
                )
            ).size(),
            (
                self.image_channels // self.image_channels,
                (self.image_size // self.patch_size) ** 2,
                self.dimension,
            ),
        )

        self.assertEqual(
            self.pathEmbedding1(
                torch.randn(
                    (
                        self.image_channels // self.image_channels,
                        self.image_channels,
                        self.image_size,
                        self.image_size,
                    )
                )
            ).size(),
            (
                self.image_channels // self.image_channels,
                (self.image_size // self.patch_size) ** 2,
                (self.patch_size**2) * self.image_channels,
            ),
        )

    def test_scaled_dot_product_func(self):
        self.assertEqual(
            self.attention.size(),
            (1, self.nheads, self.number_of_patch_size, self.dimension // self.nheads),
        )

    def test_multihead_attention(self):
        self.assertEqual(
            self.multihead_attention(
                torch.randn((1, self.number_of_patch_size, self.dimension))
            ).size(),
            (1, self.number_of_patch_size, self.dimension),
        )


if __name__ == "__main__":
    unittest.main()
