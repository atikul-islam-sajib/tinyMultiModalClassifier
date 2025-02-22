import os
import sys
import torch
import unittest
import torch.nn as nn

sys.path.append("./src/")

from utils import config_files
from patch_embedding import PatchEmbedding


class UnitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.image_channels = config_files()["patchEmbeddings"]["channels"]
        self.image_size = config_files()["patchEmbeddings"]["image_size"]
        self.patch_size = config_files()["patchEmbeddings"]["patch_size"]
        self.dimension = config_files()["patchEmbeddings"]["dimension"]

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


if __name__ == "__main__":
    unittest.main()
