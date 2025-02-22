import os
import sys
import yaml
import joblib
import torch
import torch.nn as nn

sys.path.append("/src/")


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
