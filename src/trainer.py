import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

sys.path.append("./src/")

try:
    from helper import helper
    from utils import config_files, device_init, dump_file, load_file
    from loss_functon import LossFunction
    from multi_modal_clf import MultiModalClassifier
except ImportError as e:
    print(f"Module Import Error: {e}")
    sys.exit(1)


class Trainer:
    def __init__(
        self,
        model=None,
        epochs: int = 100,
        lr: float = 2e-4,
        beta1: float = 0.5,
        beta2: float = 0.999,
        momentum: float = 0.95,
        step_size: int = 20,
        gamma: float = 0.75,
        l1_lambda: float = 0.01,
        l2_lambda: float = 0.01,
        device: str = "cuda",
        adam: bool = True,
        SGD: bool = False,
        l1_regularization: bool = False,
        l2_regularization: bool = False,
        lr_scheduler: bool = False,
        verbose: bool = True,
        mlflow: bool = False,
    ):
        try:
            self.model = model
            self.epochs = epochs
            self.lr = lr
            self.beta1 = beta1
            self.beta2 = beta2
            self.momentum = momentum
            self.step_size = step_size
            self.gamma = gamma
            self.l1_lambda = l1_lambda
            self.l2_lambda = l2_lambda
            self.adam = adam
            self.SGD = SGD
            self.device = device
            self.l1_regularization = l1_regularization
            self.l2_regularization = l2_regularization
            self.lr_scheduler = lr_scheduler
            self.verbose = verbose
            self.mlflow = mlflow

            self.device = device_init(device=self.device)

            self.init = helper(
                model=self.model,
                lr=self.lr,
                beta1=self.beta1,
                beta2=self.beta2,
                momentum=self.momentum,
                weight_decay=self.l2_lambda,
                adam=self.adam,
                SGD=self.SGD,
            )
            try:
                self.train_dataloader = self.init["train_dataloader"]
                self.test_dataloader = self.init["test_dataloader"]
                assert (
                    self.train_dataloader.__class__ == torch.utils.data.DataLoader
                ), "Train_dataloader is not a valid DataLoader"
                assert (
                    self.test_dataloader.__class__ == torch.utils.data.DataLoader
                ), "Test_dataloader is not a valid DataLoader"
            except KeyError as e:
                print(f"DataLoader Initialization Error: Missing key {e}")
                sys.exit(1)

            try:
                self.model = self.init["model"]
                assert (
                    self.model.__class__ == MultiModalClassifier
                ), "Model must be an instance of MultiModalClassifier"
            except KeyError:
                print(
                    "Model Initialization Error: 'model' key missing from helper return dictionary"
                )
                sys.exit(1)
            except AssertionError as e:
                print(e)
                sys.exit(1)

            try:
                self.optimizer = self.init["optimizer"]
                if self.adam:
                    assert (
                        self.optimizer.__class__ == optim.Adam
                    ), "Optimizer should be Adam"
                elif self.SGD:
                    assert (
                        self.optimizer.__class__ == optim.SGD
                    ), "Optimizer should be SGD"
            except KeyError:
                print(
                    "Optimizer Initialization Error: 'optimizer' key missing from helper return dictionary"
                )
                sys.exit(1)
            except AssertionError as e:
                print(e)
                sys.exit(1)

            try:
                self.criterion = self.init["criterion"]
                assert (
                    self.criterion.__class__ == LossFunction
                ), "Criterion should be a PyTorch loss function"
            except KeyError:
                print(
                    "Criterion Initialization Error: 'criterion' key missing from helper return dictionary"
                )
                sys.exit(1)
            except AssertionError as e:
                print(e)
                sys.exit(1)

            if self.lr_scheduler:
                self.scheduler = StepLR(
                    optimizer=self.optimizer, step_size=self.step_size, gamma=self.gamma
                )

            self.model = self.model.to(self.device)
            self.criterion = self.criterion.to(self.device)

        except Exception as e:
            print(f"Unexpected Error in Trainer Initialization: {e}")
            sys.exit(1)

        self.train_models = config_files()["artifacts"]["train_models"]
        self.best_model = config_files()["artifacts"]["best_model"]
        self.metrics_path = config_files()["artifacts"]["metrics"]

        self.model_history = {
            "train_loss": [],
            "test_loss": [],
            "train_accuracy": [],
            "test_accuracy": [],
        }

        self.loss = float("inf")

    def l1_regularizer(self, model):
        if isinstance(model, MultiModalClassifier):
            return self.l1_lambda * sum(
                torch.norm(input=params, p=1) for params in model.parameters()
            )
        else:
            raise ValueError("Model must be an instance of MultiModalClassifier")

    def l2_regularizer(self):
        if isinstance(model, MultiModalClassifier):
            return self.l2_lambda * sum(
                torch.norm(input=params, p=2) for params in model.parameters()
            )
        else:
            raise ValueError("Model must be an instance of MultiModalClassifier")

    def saved_checkpoints(self, train_loss: float, epoch: int):
        if self.loss > train_loss:
            self.loss = train_loss
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "train_loss": self.loss,
                    "epoch": epoch,
                },
                os.path.join(self.best_model, "best_model.pth"),
            )
        torch.save(
            self.model.state_dict(),
            os.path.join(self.train_models, "model{}.pth".format(epoch)),
        )

    def update_train(self, **kwargs):
        predicted = kwargs["predicted"].float()
        labels = kwargs["labels"].float()

        self.optimizer.zero_grad()

        loss = self.criterion(predicted, labels)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def display_progress(
        self,
        train_loss: list,
        valid_loss: list,
        train_accuracy: list,
        valid_accuracy: list,
        kwargs: dict,
    ):
        if (
            isinstance(train_loss, list)
            and isinstance(valid_loss, list)
            and isinstance(train_accuracy, list)
            and isinstance(valid_accuracy, list)
        ):
            train_loss = np.mean(train_loss)
            valid_loss = np.mean(valid_loss)
            train_accuracy = np.mean(train_accuracy)
            valid_accuracy = np.mean(valid_accuracy)
            number_of_epochs = self.epochs
            epoch = kwargs["epochs"]

            print(
                f"Epoch [{epoch}/{number_of_epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Loss: {valid_loss:.4f} | "
                f"Train Acc: {train_accuracy:.4f} | "
                f"Valid Acc: {valid_accuracy:.4f}"
            )

    def train(self):
        for epoch in tqdm(range(self.epochs), desc="Training"):
            train_loss = []
            valid_loss = []
            train_accuracy = []
            valid_accuracy = []

            for idx, (images, texts, labels) in enumerate(self.train_dataloader):
                images = images.to(self.device)
                texts = texts.to(self.device)
                labels = labels.to(self.device)

                predicted = self.model(image=images, text=texts)

                train_loss.append(self.update_train(predicted=predicted, labels=labels))

                predicted = torch.where(predicted > 0.5, 1, 0)
                predicted = predicted.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()

                train_accuracy.append(accuracy_score(predicted, labels))

            for idx, (images, texts, labels) in enumerate(self.test_dataloader):
                images = images.to(self.device)
                texts = texts.to(self.device)
                labels = labels.to(self.device)

                predicted = self.model(image=images, text=texts)

                valid_loss.append(
                    self.criterion(predicted.float(), labels.float()).item()
                )

                predicted = torch.where(predicted > 0.5, 1, 0)
                predicted = predicted.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()

                valid_accuracy.append(accuracy_score(predicted, labels))

            if self.lr_scheduler:
                self.scheduler.step()

            try:
                self.display_progress(
                    train_loss=train_loss,
                    valid_loss=valid_loss,
                    train_accuracy=train_accuracy,
                    valid_accuracy=valid_accuracy,
                    kwargs={"epochs": epoch + 1},
                )
            except KeyError as e:
                print(f"[Error] Missing key in kwargs: {e}")
            except TypeError as e:
                print(f"[Error] Type mismatch in display_progress arguments: {e}")
            except ValueError as e:
                print(f"[Error] Invalid value encountered: {e}")
            except Exception as e:
                print(f"[Unexpected Error] {e}")

            try:
                self.saved_checkpoints(train_loss=np.mean(train_loss), epoch=epoch + 1)
            except FileNotFoundError:
                print(
                    "Error: Checkpoint directory not found. Ensure the save path exists."
                )
            except PermissionError:
                print(
                    "Error: Permission denied. Cannot write to the checkpoint directory."
                )
            except TypeError as e:
                print(f"Type Error in saved_checkpoints: {e}")
            except Exception as e:
                print(f"Unexpected Error in saving checkpoint: {e}")

            self.model_history["train_loss"].append(np.mean(train_loss))
            self.model_history["train_accuracy"].append(np.mean(train_accuracy))
            self.model_history["test_loss"].append(np.mean(valid_loss))
            self.model_history["test_accuracy"].append(np.mean(valid_accuracy))

        dump_file(
            value=self.model_history,
            filename=os.path.join(self.metrics_path, "history.pkl"),
        )

    @staticmethod
    def display_history():
        metrics_path = config_files()["artifacts"]["metrics"]
        history = load_file(filename=os.path.join(metrics_path, "history.pkl"))
        if history is not None:
            _, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True)

            axes[0, 0].plot(history["train_loss"], label="Train Loss")
            axes[0, 0].plot(history["test_loss"], label="Test Loss")
            axes[0, 0].set_title("Loss")
            axes[0, 0].set_xlabel("Epochs")
            axes[0, 0].legend()

            axes[0, 1].plot(history["train_accuracy"], label="Train Accuracy")
            axes[0, 1].plot(history["test_accuracy"], label="Test Accuracy")
            axes[0, 1].set_title("Accuracy")
            axes[0, 1].set_xlabel("Epochs")
            axes[0, 1].legend()

            axes[1, 0].axis("off")
            axes[1, 1].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(metrics_path, "history.png"))
            plt.show()
            print("History saved as 'history.png' in the metrics folder".capitalize())
        else:
            print("No history found".capitalize())


if __name__ == "__main__":
    model = config_files()["trainer"]["model"]
    epochs = config_files()["trainer"]["epochs"]
    lr = float(config_files()["trainer"]["lr"])
    beta1 = float(config_files()["trainer"]["beta1"])
    beta2 = float(config_files()["trainer"]["beta2"])
    momentum = config_files()["trainer"]["momentum"]
    step_size = config_files()["trainer"]["step_size"]
    gamma = config_files()["trainer"]["gamma"]
    l1_lambda = config_files()["trainer"]["l1_lambda"]
    l2_lambda = config_files()["trainer"]["l2_lambda"]
    device = config_files()["trainer"]["device"]
    adam = config_files()["trainer"]["adam"]
    SGD = config_files()["trainer"]["SGD"]
    l1_regularization = config_files()["trainer"]["l1_regularization"]
    l2_regularization = config_files()["trainer"]["l2_regularization"]
    lr_scheduler = config_files()["trainer"]["lr_scheduler"]
    verbose = config_files()["trainer"]["verbose"]
    mlflow = config_files()["trainer"]["mlflow"]

    parser = argparse.ArgumentParser(description="Train the Multimodal Model".title())
    parser.add_argument("--model", type=str, default=model, help="Model to train")
    parser.add_argument(
        "--epochs", type=int, default=epochs, help="Number of epochs to train"
    )
    parser.add_argument("--lr", type=float, default=lr, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=beta1, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=beta2, help="Adam beta2")
    parser.add_argument("--momentum", type=float, default=momentum, help="SGD momentum")
    parser.add_argument(
        "--step_size", type=int, default=step_size, help="Step size for lr scheduler"
    )
    parser.add_argument(
        "--gamma", type=float, default=gamma, help="Gamma for lr scheduler"
    )
    parser.add_argument(
        "--l1_lambda", type=float, default=l1_lambda, help="L1 regularization lambda"
    )
    parser.add_argument(
        "--l2_lambda", type=float, default=l2_lambda, help="L2 regularization lambda"
    )
    parser.add_argument(
        "--l1_regularization",
        type=float,
        default=l1_regularization,
        help="L1 regularization",
    )
    parser.add_argument(
        "--l2_regularization",
        type=float,
        default=l2_regularization,
        help="L2 regularization",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=bool,
        default=lr_scheduler,
        help="Use learning rate scheduler",
    )
    parser.add_argument(
        "--verbose", type=bool, default=verbose, help="Display progress"
    )
    parser.add_argument(
        "--mlflow", type=bool, default=mlflow, help="Enable MLflow tracking"
    )
    parser.add_argument(
        "--device", type=str, default=device, help="Device to use (cpu or cuda)"
    )
    parser.add_argument("--adam", type=bool, default=adam, help="Use Adam optimizer")
    parser.add_argument("--SGD", type=bool, default=SGD, help="Use SGD optimizer")

    args = parser.parse_args()

    trainer = Trainer(
        model=None,
        epochs=args.epochs,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        momentum=args.momentum,
        step_size=args.step_size,
        gamma=args.gamma,
        l1_lambda=args.l1_lambda,
        l2_lambda=args.l2_lambda,
        device=args.device,
        adam=args.adam,
        SGD=args.SGD,
        l1_regularization=args.l1_regularization,
        l2_regularization=args.l2_regularization,
        lr_scheduler=args.lr_scheduler,
        verbose=args.verbose,
        mlflow=args.mlflow,
    )

    trainer.train()
    Trainer.display_history()
