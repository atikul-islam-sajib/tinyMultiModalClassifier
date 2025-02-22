import os
import sys
import yaml
import torch
import joblib
import torch.nn as nn

sys.path.append("/src/")


def config_files():
    with open("./config.yml", "r") as config_file:
        return yaml.safe_load(config_file)


def dump_file(value: None, filename: None):
    if (value is not None) and (filename is not None):
        joblib.dump(value=value, filename=filename)
    else:
        print("Error: 'value' and 'filename' must be provided.".capitalize())


def load_file(filename: None):
    if filename is not None:
        return joblib.load(filename=filename)
    else:
        print("Error: 'filename' must be provided.".capitalize())


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


def device_init(device: str = "cuda"):
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        return torch.device("cpu")
