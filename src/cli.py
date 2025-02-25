import os
import sys
import argparse

sys.path.append("./src/")

from utils import config_files
from dataloader import Loader


def cli():
    nheads = config_files()["transfomerEncoderBlock"]["nheads"]
    dropout = config_files()["transfomerEncoderBlock"]["dropout"]
    batch_size = config_files()["dataloader"]["batch_size"]
    image_size = config_files()["patchEmbeddings"]["image_size"]
    patch_size = config_files()["patchEmbeddings"]["patch_size"]
    activation = config_files()["transfomerEncoderBlock"]["activation"]
    dimension = config_files()["patchEmbeddings"]["dimension"]
    image_channels = config_files()["patchEmbeddings"]["channels"]
    num_encoder_layers = config_files()["transfomerEncoderBlock"]["num_encoder_layers"]
    dimension_feedforward = config_files()["transfomerEncoderBlock"][
        "dimension_feedforward"
    ]
    layer_norm_eps = float(config_files()["transfomerEncoderBlock"]["layer_norm_eps"])

    parser = argparse.ArgumentParser(
        description="Multi-Modal Classifier Training and Testing CLI".title()
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=config_files()["patchEmbeddings"]["channels"],
        help="Number of channels in the images",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config_files()["dataloader"]["batch_size"],
        help="Batch size for the dataloader",
    )
    parser.add_argument(
        "--split_size",
        type=float,
        default=config_files()["dataloader"]["split_size"],
        help="Split size for the train and test datasets",
    )
    parser.add_argument(
        "--image_channels",
        type=int,
        default=image_channels,
        help="Image channels to transform".capitalize(),
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=image_size,
        help="Image size to transform".capitalize(),
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=patch_size,
        help="Patch size for the transformer".capitalize(),
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=dimension,
        help="Dimension for the transformer".capitalize(),
    )
    parser.add_argument(
        "--nheads",
        type=int,
        default=nheads,
        help="Number of heads for the multi-head attention".capitalize(),
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=activation,
        help="Activation function for the transformer".capitalize(),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=dropout,
        help="Dropout probability for the transformer".capitalize(),
    )
    parser.add_argument(
        "--num_encoder_layers",
        type=int,
        default=num_encoder_layers,
        help="Number of layers in the transformer".capitalize(),
    )
    parser.add_argument(
        "--dimension_feedforward",
        type=int,
        default=dimension_feedforward,
        help="Dimension for the feedforward layer".capitalize(),
    )
    parser.add_argument(
        "--layer_norm_eps",
        type=float,
        default=layer_norm_eps,
        help="Layer normalization epsilon".capitalize(),
    )
    parser.add_argument(
        "--train",
        action="store_true",
        default="Train CLI for the Multi Modal Classifier".capitalize(),
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default="Test CLI for the Multi Modal Classifier".capitalize(),
    )

    args = parser.parse_args()

    if args.train:
        loader = Loader(
            channels=args.channels,
            image_size=args.image_size,
            batch_size=args.batch_size,
            split_size=args.split_size,
        )
        loader.unzip_image_dataset()
        loader.create_dataloader()

        Loader.display_images()
        Loader.details_dataset()
        
        


if __name__ == "__main__":
    cli()
