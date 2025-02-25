import os
import sys
import torch
import unittest
import torch.nn as nn
import torch.optim as optim

sys.path.append("./src/")

from utils import config_files
from patch_embedding import PatchEmbedding
from scaled_dot_product import scaled_dot_product
from multihead_attention_layer import MultiHeadAttentionLayer
from feed_forward_network import FeedForwardNeuralNetwork
from layer_normalization import LayerNormalization
from transformer_encoder_block import TransformerEncoderBlock
from transformer_encoder import TransformerEncoder
from loss_functon import LossFunction
from ViT import VisionTransformer
from text_transformer import TextTransformerEncoder
from multi_modal_clf import MultiModalClassifier
from helper import helper


class UnitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.image_channels = config_files()["patchEmbeddings"]["channels"]
        self.image_size = config_files()["patchEmbeddings"]["image_size"]
        self.patch_size = config_files()["patchEmbeddings"]["patch_size"]
        self.dimension = config_files()["patchEmbeddings"]["dimension"]
        self.nheads = config_files()["transfomerEncoderBlock"]["nheads"]
        self.activation = config_files()["transfomerEncoderBlock"]["activation"]
        self.dropout = config_files()["transfomerEncoderBlock"]["dropout"]
        self.num_encoder_layers = config_files()["transfomerEncoderBlock"][
            "num_encoder_layers"
        ]
        self.dimension_feedforward = config_files()["transfomerEncoderBlock"][
            "dimension_feedforward"
        ]
        self.layer_norm_eps = float(
            config_files()["transfomerEncoderBlock"]["layer_norm_eps"]
        )
        self.lr = float(config_files()["trainer"]["lr"])
        self.batch_size = config_files()["dataloader"]["batch_size"]
        self.epochs = config_files()["trainer"]["epochs"]
        self.weight_decay = float(config_files()["trainer"]["weight_decay"])
        self.beta1 = config_files()["trainer"]["beta1"]
        self.beta2 = config_files()["trainer"]["beta2"]
        self.momentum = config_files()["trainer"]["momentum"]
        self.adam = config_files()["trainer"]["adam"]
        self.SGD = config_files()["trainer"]["SGD"]

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
        self.FFNN = FeedForwardNeuralNetwork(
            in_features=self.dimension, out_features=4 * self.dimension
        )
        self.layer_normalization = LayerNormalization(dimension=self.dimension)
        self.transformer_encoder_block = TransformerEncoderBlock(
            dimension=self.dimension,
            nheads=self.nheads,
            dim_feedforward=self.dimension * 4,
            dropout=self.dropout,
            activation=self.activation,
        )
        self.transformer_encoder = TransformerEncoder(
            dimension=self.dimension,
            nheads=self.nheads,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dimension * 4,
            dropout=self.dropout,
            activation=self.activation,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.criterion = LossFunction(
            loss_name="BCELoss",
            reduction="mean",
        )
        self.vision_transformer = VisionTransformer(
            channels=self.image_channels,
            patch_size=self.patch_size,
            image_size=self.image_size,
            dimension=self.dimension,
            nheads=self.nheads,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dimension_feedforward,
            dropout=self.dropout,
            layer_norm_eps=self.layer_norm_eps,
            activation=self.activation,
        )
        self.text_transformer = TextTransformerEncoder(
            dimension=self.dimension,
            nheads=self.nheads,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dimension_feedforward,
            dropout=self.dropout,
            layer_norm_eps=self.layer_norm_eps,
            activation=self.activation,
        )

        self.classifier = MultiModalClassifier(
            channels=self.image_channels,
            patch_size=self.patch_size,
            image_size=self.image_size,
            dimension=self.dimension,
            nheads=self.nheads,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dimension_feedforward,
            dropout=self.dropout,
            layer_norm_eps=self.layer_norm_eps,
            activation=self.activation,
        )

        self.init = helper(
            model = None,
            lr = self.lr,
            beta1 = self.beta1,
            beta2 = self.beta2,
            momentum = self.momentum,
            weight_decay = self.weight_decay,
            adam = self.adam,
            SGD = self.SGD,
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

    def test_FFNN(self):
        self.assertEqual(
            self.FFNN(
                torch.randn((1, self.number_of_patch_size, self.dimension))
            ).size(),
            (1, self.number_of_patch_size, self.dimension),
        )

    def test_layer_normalization(self):
        self.assertEqual(
            self.layer_normalization(
                torch.randn((1, self.number_of_patch_size, self.dimension))
            ).size(),
            (1, self.number_of_patch_size, self.dimension),
        )

    def test_transformer_encoder_block(self):
        self.assertEqual(
            self.transformer_encoder_block(
                torch.randn((1, self.number_of_patch_size, self.dimension))
            ).size(),
            (1, self.number_of_patch_size, self.dimension),
        )

    def test_transformer_encoder(self):
        self.assertEqual(
            self.transformer_encoder(
                torch.randn((1, self.number_of_patch_size, self.dimension))
            ).size(),
            (1, self.number_of_patch_size, self.dimension),
        )

    def test_vision_transformer(self):
        self.assertEqual(
            self.vision_transformer(
                torch.randn((1, self.image_channels, self.image_size, self.image_size))
            ).size(),
            (1, (self.image_size // self.patch_size) ** 2, self.dimension),
        )

    def test_loss_function(self):
        y_true = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float)
        y_pred = torch.tensor([0.9, 0.1, 0.8, 0.2])
        self.assertEqual(self.criterion(y_true, y_pred).item(), 15.0)
        self.assertIsInstance(self.criterion(y_true, y_pred), torch.Tensor)

    def test_text_transformer_encoder(self):
        textual_data = torch.randint(
            0,
            (self.image_size // self.patch_size) ** 2,
            (1, (self.image_size // self.patch_size) ** 2),
        )
        self.assertEqual(
            self.text_transformer(textual_data).size(),
            (1, (self.image_size // self.patch_size) ** 2, self.dimension),
        )
        self.assertIsInstance(self.text_transformer(textual_data), torch.Tensor)

    # def test_multimodal_classifier_without_fusion(self):
    #     number_of_patches = (self.image_size // self.patch_size) ** 2
    #     number_of_sequences = (self.image_size // self.patch_size) ** 2

    #     images = torch.randn(1, self.image_channels, self.image_size, self.image_size)
    #     texts = torch.randint(0, number_of_sequences, (1, number_of_sequences))

    #     outputs = self.classifier(image=images, text=texts)

    #     self.assertEqual(
    #         outputs["image_features"].size(), (1, number_of_patches, self.dimension)
    #     )
    #     self.assertEqual(
    #         outputs["text_features"].size(), (1, number_of_sequences, self.dimension)
    #     )

    def test_helper_function(self):
        assert self.init["model"].__class__ == MultiModalClassifier
        assert self.init["optimizer"].__class__ == optim.Adam
        assert self.init["criterion"].__class__ == LossFunction
        assert self.init["train_dataloader"].__class__ == torch.utils.data.DataLoader
        assert self.init["test_dataloader"].__class__ == torch.utils.data.DataLoader


if __name__ == "__main__":
    unittest.main()
