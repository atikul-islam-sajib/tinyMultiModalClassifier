import os
import sys
import torch
import argparse
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.append("./src/")


class Loader:
    def __init__(
        self,
        channels: int = 3,
        image_size: int = 128,
        batch_size: int = 4,
        split_size: float = 0.25,
    ):
        self.channels = channels
        self.image_size = image_size
        self.batch_size = batch_size
        self.split_size = split_size

    def image_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.CenterCrop((self.image_size, self.image_size)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def preprocess_csv_file(self):
        pass

    def unzip_image_dataset(self):
        pass

    def split_dataset(self, X: list, y: list):
        pass

    def extracted_text(self):
        pass

    def extracted_image_features(self):
        pass

    def create_dataloader(self):
        pass


if __name__ == "__main__":
    pass
