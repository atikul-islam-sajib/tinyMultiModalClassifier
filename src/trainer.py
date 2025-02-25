import os
import sys
import torch
import torch.nn as nn

sys.path.append("./src/")


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
    pass
