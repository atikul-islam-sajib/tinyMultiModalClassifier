import os
import sys
import torch.nn as nn
import torch.optim as optim

sys.path.append("./src/")

from loss_functon import LossFunction
from utils import config_files, load_file
from multi_modal_clf import MultiModalClassifier


def load_dataloader():
    processed_data_path = config_files()["artifacts"]["processed_data_path"]
    if os.path.exists(processed_data_path):
        train_dataloader_path = os.path.join(
            processed_data_path, "train_dataloader.pkl"
        )
        valid_dataloader_path = os.path.join(processed_data_path, "test_dataloader.pkl")

        train_dataloader = load_file(filename=train_dataloader_path)
        valid_dataloader = load_file(filename=valid_dataloader_path)

        return {
            "train_dataloader": train_dataloader,
            "valid_dataloader": valid_dataloader,
        }


def helper(**kwargs):
    model = kwargs["model"]
    lr: float = kwargs["lr"]
    beta1: float = kwargs["beta1"]
    beta2: float = kwargs["beta2"]
    momentum: float = kwargs["momentum"]
    weight_decay: float = kwargs["weight_decay"]
    adam: bool = kwargs["adam"]
    SGD: bool = kwargs["SGD"]

    if model is None:
        nheads = config_files()["transfomerEncoderBlock"]["nheads"]
        dropout = config_files()["transfomerEncoderBlock"]["dropout"]
        image_size = config_files()["patchEmbeddings"]["image_size"]
        patch_size = config_files()["patchEmbeddings"]["patch_size"]
        activation = config_files()["transfomerEncoderBlock"]["activation"]
        dimension = config_files()["patchEmbeddings"]["dimension"]
        image_channels = config_files()["patchEmbeddings"]["channels"]
        num_encoder_layers = config_files()["transfomerEncoderBlock"][
            "num_encoder_layers"
        ]
        dimension_feedforward = config_files()["transfomerEncoderBlock"][
            "dimension_feedforward"
        ]
        layer_norm_eps = float(
            config_files()["transfomerEncoderBlock"]["layer_norm_eps"]
        )

        classifier = MultiModalClassifier(
            channels=image_channels,
            patch_size=patch_size,
            image_size=image_size,
            nheads=nheads,
            dropout=dropout,
            activation=activation,
            dimension=dimension,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dimension_feedforward,
            layer_norm_eps=layer_norm_eps,
        )

    else:
        classifier = model

    if adam:
        optimizer = optim.Adam(
            params=classifier.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
        )
    elif SGD:
        optimizer = optim.SGD(params=classifier.parameters(), lr=lr, momentum=momentum)
    else:
        raise ValueError("Optimizer not supported".capitalize())

    criterion = LossFunction(loss_name="BCEWithLogitsLoss", reduction="mean")

    try:
        dataset = load_dataloader()
    except Exception as e:
        print(f"Error while loading dataset: {e}")
        sys.exit(1)

    return {
        "train_dataloader": dataset["train_dataloader"],
        "test_dataloader": dataset["valid_dataloader"],
        "model": classifier,
        "optimizer": optimizer,
        "criterion": criterion,
    }


if __name__ == "__main__":
    init = helper(
        model = None,
        lr = float(config_files()["trainer"]["lr"]),
        beta1 = config_files()["trainer"]["beta1"],
        beta2 = config_files()["trainer"]["beta2"],
        momentum = config_files()["trainer"]["momentum"],
        weight_decay = float(config_files()["trainer"]["weight_decay"]),
        adam = config_files()["trainer"]["adam"],
        SGD = config_files()["trainer"]["SGD"],
    )

    assert init["model"].__class__ == MultiModalClassifier
    assert init["optimizer"].__class__ == optim.Adam
