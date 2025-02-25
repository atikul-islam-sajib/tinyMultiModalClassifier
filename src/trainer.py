import os
import sys
import torch
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

    def l1_regularizer(self):
        pass

    def l2_regularizer(self):
        pass

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
    lr = config_files()["trainer"]["lr"]
    beta1 = config_files()["trainer"]["beta1"]
    beta2 = config_files()["trainer"]["beta2"]
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

    trainer = Trainer(
        model=None,
        epochs=epochs,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        momentum=momentum,
        step_size=step_size,
        gamma=gamma,
        l1_lambda=l1_lambda,
        l2_lambda=l2_lambda,
        device=device,
        adam=adam,
        SGD=SGD,
        l1_regularization=l1_regularization,
        l2_regularization=l2_regularization,
        lr_scheduler=lr_scheduler,
        verbose=verbose,
        mlflow=mlflow,
    )
