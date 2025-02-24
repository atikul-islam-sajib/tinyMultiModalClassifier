import os
import cv2
import sys
import math
import torch
import zipfile
import argparse
import pandas as pd
import numpy as np
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from textwrap import fill
from sklearn.model_selection import train_test_split

sys.path.append("./src/")

from utils import config_files, text_preprocessing, dump_file, load_file


class Loader:
    def __init__(
        self,
        channels: int = 3,
        image_size: int = 128,
        patch_size: int = 16,
        batch_size: int = 4,
        split_size: float = 0.25,
    ):
        self.channels = channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.split_size = split_size

        self.vocabulary = {"<UNK>": 0}
        self.images_data = list()
        self.labels_data = list()
        self.textual_data = list()
        self.text_to_sequence = list()
        self.sequences = list()

        self.sequence_length = (self.image_size // self.patch_size) ** 2

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
        dataframe_path = os.path.join(
            config_files()["artifacts"]["raw_data_path"], "image_labels_reports.csv"
        )
        df = pd.read_csv(dataframe_path)
        if ("text" in df.columns) and ("label" in df.columns) and ("img" in df.columns):
            labels = df.loc[:, "label"]
            reports = df.loc[:, "text"]
            images = df.loc[:, "img"]

            df["text"] = df["text"].apply(text_preprocessing)
            reports = reports.apply(text_preprocessing)

            return {
                "labels": labels,
                "reports": reports,
                "images": images,
                "dataframe": df,
            }
        else:
            raise ValueError(
                "The 'text' and 'labels' columns are missing in the CSV file.".capitalize()
            )

    def create_vocabularies(self, instance):
        for word in instance.split(" "):
            if word not in self.vocabulary:
                self.vocabulary[word] = len(self.vocabulary)

    def unzip_image_dataset(self):
        if os.path.exists(config_files()["artifacts"]["raw_data_path"]):
            image_data_path = os.path.join(
                config_files()["artifacts"]["raw_data_path"], "image_dataset.zip"
            )
            processed_data_path = config_files()["artifacts"]["processed_data_path"]

            with zipfile.ZipFile(file=image_data_path, mode="r") as zip_file:
                zip_file.extractall(
                    path=os.path.join(processed_data_path, "image_dataset")
                )

            print(
                "Image dataset unzipped successfully in the folder {}".capitalize().format(
                    processed_data_path
                )
            )
        else:
            raise FileNotFoundError("Could not extract image dataset".capitalize())

    def create_sequences(self, instance, sequence_length: int):
        sequence = [
            self.vocabulary.get(word, self.vocabulary["<UNK>"])
            for word in instance.split()
        ]
        if len(sequence) > sequence_length:
            sequence = sequence[:sequence_length]

        elif len(sequence) < sequence_length:
            sequence.extend(
                [self.vocabulary["<UNK>"]] * (sequence_length - len(sequence))
            )

        assert (
            len(sequence) == sequence_length
        ), f"Error: Sequence length is {len(sequence)} instead of {sequence_length}"

        self.sequences.append(sequence)

        return sequence

    def extracted_image_and_text_features(self):
        dataset = self.preprocess_csv_file()
        images = dataset["images"]
        labels = dataset["labels"]
        reports = dataset["reports"]
        dataframe = dataset["dataframe"]

        try:
            reports.apply(self.create_vocabularies)

            pd.DataFrame(
                list(self.vocabulary.items()), columns=["vocabulary", "index"]
            ).to_csv(
                os.path.join(
                    config_files()["artifacts"]["processed_data_path"], "vocabulary.csv"
                )
            )
        except Exception as e:
            print(f"Error occurred while creating vocabularies: {e}")
            sys.exit(1)

        dataframe["sequences"] = reports.apply(
            self.create_sequences, sequence_length=self.sequence_length
        )

        all_image_path = os.path.join(
            config_files()["artifacts"]["processed_data_path"], "image_dataset"
        )

        for image in tqdm(os.listdir(all_image_path), desc="Processing Images"):
            try:
                if image not in images.values.tolist():
                    print(f"Image not found in dataset: {image}")
                    continue

                try:
                    text = dataframe.loc[dataframe["img"] == image, "text"].values[0]
                    label = dataframe.loc[dataframe["img"] == image, "label"].values[0]
                    sequences = dataframe.loc[
                        dataframe["img"] == image, "sequences"
                    ].values[0]
                except IndexError:
                    print(f"Missing data for image: {image}")
                    continue

                single_image_path = os.path.join(all_image_path, image)

                if not single_image_path.lower().endswith(("jpeg", "png", "jpg")):
                    print(f"Invalid file format: {single_image_path}")
                    continue

                if not os.path.exists(single_image_path):
                    print(f"File does not exist: {single_image_path}")
                    continue

                try:
                    image_data = cv2.imread(single_image_path)
                    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
                    if image_data is None:
                        raise ValueError(
                            f"Corrupted or unreadable image: {single_image_path}"
                        )

                    image_data = Image.fromarray(image_data)
                    image_data = self.image_transform()(image_data)

                    if not isinstance(image_data, torch.Tensor):
                        raise TypeError(
                            f"Expected torch.Tensor but got {type(image_data)} for {image}"
                        )

                    self.images_data.append(image_data)
                    self.labels_data.append(label)
                    self.textual_data.append(text)
                    self.text_to_sequence.append(sequences)

                except Exception as e:
                    print(f"Error processing image '{image}': {e}")
                    continue

            except Exception as e:
                print(f"Unexpected error processing file '{image}': {e}")

        assert (
            len(self.images_data)
            == len(self.labels_data)
            == len(self.textual_data)
            == len(self.text_to_sequence)
        ), "Mismatch: 'Image data', 'labels', 'text', and 'text to sequence' are not equal"

        try:
            self.labels_data = torch.tensor(self.labels_data, dtype=torch.long)
            self.text_to_sequence = torch.tensor(
                self.text_to_sequence, dtype=torch.long
            )
        except Exception as e:
            print("Tensor conversion failed: {e}")

        return {
            "images": self.images_data,
            "labels": self.labels_data,
            "text_to_sequence": self.text_to_sequence,
        }

    def create_dataloader(self):
        try:
            dataset = self.extracted_image_and_text_features()
            images = dataset["images"]
            labels = dataset["labels"]
            text_to_sequence = dataset["text_to_sequence"]

            test_image_portion = int(len(images) * self.split_size)
            test_labels_portion = int(len(labels) * self.split_size)
            test_texts_portion = int(text_to_sequence.size(0) * self.split_size)

            train_images = images[:-test_image_portion]
            train_labels = labels[:-test_labels_portion]
            train_texts = text_to_sequence[:-test_texts_portion]

            test_images = images[-test_image_portion:]
            test_labels = labels[-test_labels_portion:]
            test_texts = text_to_sequence[-test_texts_portion:]

            if (
                len(train_images) == 0
                or len(train_labels) == 0
                or len(train_texts) == 0
            ):
                raise ValueError("Train dataset is empty! Check split size.")
            if len(test_images) == 0 or len(test_labels) == 0 or len(test_texts) == 0:
                raise ValueError("Test dataset is empty! Check split size.")

            train_dataloader = DataLoader(
                dataset=list(zip(train_images, train_texts, train_labels)),
                batch_size=self.batch_size,
                shuffle=True,
            )
            test_dataloader = DataLoader(
                dataset=list(zip(test_images, test_texts, test_labels)),
                batch_size=self.batch_size,
                shuffle=False,
            )

            try:
                for filename, value in [
                    ("train_dataloader.pkl", train_dataloader),
                    ("test_dataloader.pkl", test_dataloader),
                ]:
                    dump_file(
                        value=value,
                        filename=os.path.join(
                            config_files()["artifacts"]["processed_data_path"],
                            filename,
                        ),
                    )

                print(
                    "Dataloaders created successfully in the folder {}".capitalize().format(
                        config_files()["artifacts"]["processed_data_path"]
                    )
                )

            except StopIteration:
                raise RuntimeError(
                    "Train dataloader is empty. Check data loading logic."
                )

            return train_dataloader, test_dataloader

        except Exception as e:
            print(f"Error in create_dataloader: {e}")
            return None, None

    @staticmethod
    def details_dataset():
        train_dataloader = load_file(
            filename=os.path.join(
                config_files()["artifacts"]["processed_data_path"],
                "train_dataloader.pkl",
            )
        )
        test_dataloader = load_file(
            filename=os.path.join(
                config_files()["artifacts"]["processed_data_path"],
                "test_dataloader.pkl",
            )
        )

        train_images, _, train_labels = next(iter(train_dataloader))
        _, test_sequences, _ = next(iter(test_dataloader))

        total_train_dataset = sum(image.size(0) for image, _, _ in train_dataloader)
        total_test_dataset = sum(image.size(0) for image, _, _ in test_dataloader)

        pd.DataFrame(
            {
                "Dataset": ["Train", "Test"],
                "Size": [total_train_dataset, total_test_dataset],
                "Number of Batches": [len(train_dataloader), len(test_dataloader)],
                "Image Size": str([train_images.size()]),
                "Sequence Size": str([test_sequences.size()]),
                "Label Size": str([train_labels.size()]),
                "Label Type": str([train_labels.dtype]),
                "Text Type": str([test_sequences.dtype]),
            }
        ).to_csv(
            os.path.join(
                config_files()["artifacts"]["processed_data_path"],
                "dataset_details.csv",
            )
        )
        print(
            "Dataset details saved successfully in the folder {}".capitalize().format(
                config_files()["artifacts"]["processed_data_path"]
            )
        )

    @staticmethod
    def display_images():
        try:
            train_dataloader = load_file(
                filename=os.path.join(
                    config_files()["artifacts"]["processed_data_path"],
                    "train_dataloader.pkl",
                )
            )
            vocabularies = pd.read_csv(
                os.path.join(
                    config_files()["artifacts"]["processed_data_path"], "vocabulary.csv"
                )
            )
            vocabularies["index"] = vocabularies["index"].astype(int)

            images, texts, labels = next(iter(train_dataloader))

            num_images = images.size(0)
            num_rows = int(math.sqrt(num_images))
            num_cols = math.ceil(num_images / num_rows)

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
            axes = axes.flatten()

            for index, (image, ax) in enumerate(zip(images, axes)):
                image = image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
                image = (image - image.min()) / (image.max() - image.min())
                label = labels[index].item()
                text_sequences = texts[index].detach().cpu().numpy().tolist()
                words = vocabularies[vocabularies["index"].isin(text_sequences)][
                    "vocabulary"
                ].tolist()
                medical_report = " ".join(words).replace("<UNK>", "").strip()
                wrapped_report = fill(medical_report, width=30)

                title_text = f"Label: {label}\nReport: {wrapped_report}"

                ax.set_title(title_text, fontsize=9, loc="center")
                ax.imshow(image)
                ax.axis("off")

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error in display_images: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DataLoader for the multi model classification".title()
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=config_files()["patchEmbeddings"]["channels"],
        help="Number of channels in the images",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=config_files()["patchEmbeddings"]["image_size"],
        help="Image size to resize the images",
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
    args = parser.parse_args()

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
