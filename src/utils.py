import os
import re
import sys
import yaml
import math
import nltk
import json
import torch
import joblib
import warnings
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from textwrap import fill
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from torchvision import transforms
from torch.utils.data import DataLoader

nltk.download("stopwords")

sys.path.append("/src/")

stop_words = set(stopwords.words("english"))


def config_files():
    with open("./config.yml", "r") as config_file:
        return yaml.safe_load(config_file)


def dump_json(**kwargs):
    with open("./artifacts/metrics/metrics.json", mode="w") as json_file:
        if isinstance(kwargs, dict):
            json.dump(kwargs, json_file, indent=4)
        else:
            print("Error: 'kwargs' must be a dictionary".capitalize())


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


def clean_folders():
    TRAIN_MODELS: str = "../artifacts/checkpoints/train_models/"
    BEST_MODEL: str = "../artifacts/checkpoints/best_model/"
    METRICS_PATH: str = "../artifacts/metrics/"
    TRAIN_IMAGES: str = "../artifacts/outputs/train_images/"
    TEST_IMAGE: str = "../artifacts/outputs/test_image/"

    warnings.warn(
        "Warning! This will remove the previous files, which might be useful in the future. "
        "You may want to download the previous files or use MLflow to track and manage them. "
        "Suggestions for updating them are welcome."
    )

    for path in tqdm(
        [TRAIN_MODELS, BEST_MODEL, METRICS_PATH, TRAIN_IMAGES, TEST_IMAGE]
    ):
        for files in os.listdir(path):
            file_path = os.path.join(path, files)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error occurred while deleting {file_path}: {e}")

        print("{} folders completed".format().capitalize())


def text_preprocessing(instance):
    instance = re.sub(r'[\n\'"()]+|XXXX|x-\d{4}', "", instance)
    instance = re.sub(r"[^a-wy-zA-WY-Z\s]", "", instance)

    instance = instance.lower()

    instance = " ".join(word for word in instance.split() if word not in stop_words)

    return instance


def create_sequences(instance, vocabulary: int = 4096, sequence_length: int = 196):
    sequence = [vocabulary.get(word, vocabulary["<UNK>"]) for word in instance.split()]
    if len(sequence) > sequence_length:
        sequence = sequence[:sequence_length]

    elif len(sequence) < sequence_length:
        sequence.extend([vocabulary["<UNK>"]] * (sequence_length - len(sequence)))
    assert (
        len(sequence) == sequence_length
    ), f"Error: Sequence length is {len(sequence)} instead of {sequence_length}"


def plot_images(
    predicted: bool = False,
    device: str = "cuda",
    model=None,
    epoch: int = 1,
    dataloader: str = "train",
):
    processed_path = config_files()["artifacts"]["processed_data_path"]
    train_images_path = config_files()["artifacts"]["train_images"]
    valid_images_path = config_files()["artifacts"]["test_image"]
    saved_images_path = (
        train_images_path if dataloader == "train" else valid_images_path
    )
    try:
        train_dataloader = load_file(
            filename=os.path.join(processed_path, "train_dataloader.pkl")
        )
        valid_dataloader = load_file(
            filename=os.path.join(processed_path, "test_dataloader.pkl")
        )
        vocabularies = pd.read_csv(os.path.join(processed_path, "vocabulary.csv"))
        vocabularies["index"] = vocabularies["index"].astype(int)

        images, texts, labels = next(
            iter(train_dataloader if dataloader == "train" else valid_dataloader)
        )

        predict = model(image=images.to(device), text=texts.to(device))
        predict = torch.where(predict > 0.5, 1, 0)
        predict = predict.detach().cpu().numpy()

        max_imgs = 4

        num_images = images[:max_imgs].size(0)
        num_rows = int(math.sqrt(num_images))
        num_cols = math.ceil(num_images / num_rows)

        _, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
        axes = axes.flatten()

        for index, (image, ax) in enumerate(zip(images[:max_imgs], axes)):
            image = image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            image = (image - image.min()) / (image.max() - image.min())
            label = labels[index].item()
            text_sequences = texts[index].detach().cpu().numpy().tolist()
            words = vocabularies[vocabularies["index"].isin(text_sequences)][
                "vocabulary"
            ].tolist()
            medical_report = " ".join(words).replace("<UNK>", "").strip()
            wrapped_report = fill(medical_report, width=30)

            if predicted:
                title_text = f"**Label**: {label}\n**Predicted**: {predict[index]}\nReport: {wrapped_report}".title()
            else:
                title_text = f"Label: {label}\nReport: {wrapped_report}"

            ax.set_title(title_text, fontsize=9, loc="center")
            ax.imshow(image)
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(saved_images_path, "image{}.png".format(epoch)))
        plt.close()

    except Exception as e:
        print(f"Error in display_images: {e}")
