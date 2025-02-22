import os
import sys
import math
import torch
import argparse

sys.path.append("./src/")

from utils import config_files


def scaled_dot_product(query: torch.Tensor, key: torch.Tensor, values: torch.Tensor):
    if (
        isinstance(query, torch.Tensor)
        and isinstance(key, torch.Tensor)
        and isinstance(values, torch.Tensor)
    ):
        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.size(-1))
        attention_weights = torch.softmax(logits, dim=-1)
        attention_output = torch.matmul(attention_weights, values)
        return attention_output

    else:
        raise ValueError("All inputs must be torch tensors.".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scaled dot product for the attention".title()
    )
    image_channels = config_files()["patchEmbeddings"]["channels"]
    image_size = config_files()["patchEmbeddings"]["image_size"]
    patch_size = config_files()["patchEmbeddings"]["patch_size"]
    dimension = config_files()["patchEmbeddings"]["dimension"]
    nheads = config_files()["transfomerEncoderBlock"]["nheads"]

    number_of_patch_size = (image_size // patch_size) ** 2

    attention = scaled_dot_product(
        torch.randn((1, nheads, number_of_patch_size, dimension // nheads)),
        torch.randn((1, nheads, number_of_patch_size, dimension // nheads)),
        torch.randn((1, nheads, number_of_patch_size, dimension // nheads)),
    )
    assert attention.size() == (1, nheads, number_of_patch_size, dimension // nheads)
