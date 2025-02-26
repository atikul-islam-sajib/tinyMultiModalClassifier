import os
import sys
import torch
import warnings
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

sys.path.append("./src/")

from multi_modal_clf import MultiModalClassifier
from utils import config_files, load_file, device_init, plot_images, dump_json


class Tester:
    def __init__(
        self, model: str = "best", device: str = "cuda", plot_images: bool = False
    ):
        self.model = model
        self.device = device
        self.plot_images = plot_images

        self.accuracy = []
        self.precision = []
        self.recall = []
        self.f1_score = []

        try:
            self.device = device_init(device=self.device)

            best_model_path = config_files()["artifacts"]["best_model"]
            best_model_path = os.path.join(best_model_path, "best_model.pth")

            selected_model = self.model

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

            self.classifier = MultiModalClassifier(
                channels=image_channels,
                patch_size=patch_size,
                image_size=image_size,
                dimension=dimension,
                nheads=nheads,
                num_encoder_layers=num_encoder_layers,
                dim_feedforward=dimension_feedforward,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
                activation=activation,
            )
            self.classifier.to(self.device)

            if self.model == "best":
                try:
                    state_dict = torch.load(best_model_path, weights_only=False)
                    self.classifier.load_state_dict(state_dict["model"])
                except FileNotFoundError:
                    raise FileNotFoundError(f"Model file not found: {best_model_path}")
                except KeyError:
                    raise KeyError(
                        "The model checkpoint does not contain expected keys."
                    )
                except Exception as e:
                    raise RuntimeError(f"Error loading model: {e}")

            else:
                warnings.warn(
                    "You have selected a custom model. Make sure the path is correct!"
                )
                try:
                    state_dict = torch.load(selected_model)
                    self.classifier.load_state_dict(state_dict["model"])
                except Exception as e:
                    raise RuntimeError(f"Error loading selected model: {e}")

        except Exception as e:
            print(f"[ERROR] Model initialization failed: {e}")
            sys.exit(1)

    def model_eval(self, display_image: bool = False):
        try:
            processed_data = config_files()["artifacts"]["processed_data_path"]
            validation_data = os.path.join(processed_data, "test_dataloader.pkl")
            validation_data = load_file(filename=validation_data)

            total_predicted = []
            total_actual_labels = []

            for index, (images, texts, labels) in enumerate(validation_data):
                try:
                    images = images.to(self.device)
                    texts = texts.to(self.device)
                    labels = labels.to(self.device)

                    predicted = self.classifier(image=images, text=texts)
                    predicted = torch.where(predicted > 0.5, 1, 0)
                    predicted = predicted.detach().cpu().numpy()
                    labels = labels.detach().cpu().numpy()

                    self.accuracy.append(accuracy_score(predicted, labels))
                    self.precision.append(
                        precision_score(predicted, labels, zero_division=0)
                    )
                    self.recall.append(recall_score(predicted, labels, zero_division=0))
                    self.f1_score.append(f1_score(predicted, labels, zero_division=0))

                    total_predicted.extend(predicted)
                    total_actual_labels.extend(labels)

                except Exception as e:
                    print(f"[WARNING] Skipping batch {index} due to error: {e}")
                    continue

            if display_image:
                try:
                    plot_images(
                        predicted=True,
                        device=self.device,
                        model=self.classifier,
                        epoch=1,
                        dataloader="valid",
                    )
                except Exception as e:
                    print(f"[WARNING] Failed to display images: {e}")

            try:
                os.makedirs("./artifacts/metrics", exist_ok=True)
                dump_json(
                    accuracy=np.mean(self.accuracy),
                    precision=np.mean(self.precision),
                    recall=np.mean(self.recall),
                    f1_score=np.mean(self.f1_score),
                )
            except Exception as e:
                print(f"[ERROR] Failed to save evaluation metrics: {e}")

            try:
                conf_matrix = confusion_matrix(total_actual_labels, total_predicted)
                disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
                disp.plot()

                plt.savefig("./artifacts/metrics/confusion_matrix.png")
                plt.close()

                print("[INFO] Confusion matrix saved successfully.")

            except Exception as e:
                print(f"[ERROR] Failed to generate confusion matrix: {e}")

        except Exception as e:
            print(f"[ERROR] Model evaluation failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    model = config_files()["tester"]["model"]
    device = config_files()["tester"]["device"]
    plot_images = config_files()["tester"]["plot_images"]
    parser = argparse.ArgumentParser(description="Test MultiModalClassifier".title())
    parser.add_argument(
        "--model",
        type=str,
        default=model,
        help="Choose model to evaluate (best or path)",
    )
    parser.add_argument(
        "--device", type=str, default=device, help="Choose device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--plot_images", type=bool, default=False, help="Display predicted images"
    )
    args = parser.parse_args()
    try:
        tester = Tester(
            model=args.model, device=args.device, plot_images=args.plot_images
        )
        tester.model_eval(display_image=True)
    except Exception as e:
        print(f"[FATAL] Tester encountered a critical error: {e}")
        sys.exit(1)
    else:
        print("[INFO] MultiModalClassifier evaluation completed successfully. "
            "All files related to the test are stored in the metrics folder.")

    
