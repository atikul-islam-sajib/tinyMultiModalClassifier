import os
import sys
import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

sys.path.append("./src/")

try:
    from helper import helper
    from utils import config_files, device_init
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

            self.model = self.model.to(self.device)
            self.criterion = self.criterion.to(self.device)

        except Exception as e:
            print(f"Unexpected Error in Trainer Initialization: {e}")
            sys.exit(1)

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

    def saved_checkpoints(self):
        pass

    def update_train(self):
        pass

    def display_progress(self):
        pass

    def train(self):
        pass


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
