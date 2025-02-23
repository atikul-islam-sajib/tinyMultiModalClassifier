import torch
import torch.nn as nn


class LossFunction(nn.Module):
    def __init__(self, loss_name: str = "BCEWithLogitsLoss", reduction: str = "mean"):
        super(LossFunction, self).__init__()
        self.loss_name = loss_name
        self.reduction = reduction

        if loss_name == "BCEWithLogitsLoss":
            self.criterion = nn.BCEWithLogitsLoss(reduction=reduction)
        elif loss_name == "BCELoss":
            self.criterion = nn.BCELoss(reduction=reduction)
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")

    def forward(self, predicted: torch.Tensor, actual: torch.Tensor):
        if isinstance(predicted, torch.Tensor) and isinstance(actual, torch.Tensor):
            return self.criterion(predicted, actual)
        else:
            raise ValueError(
                "Both predicted and actual must be torch.Tensor.".capitalize()
            )


if __name__ == "__main__":
    predicted = torch.tensor(
        [2.5, -1.0, 1.5, -2.0, 2.0, 3.0, -1.5, -0.5], dtype=torch.float
    )
    actual = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0], dtype=torch.float)

    criterion = LossFunction(loss_name="BCEWithLogitsLoss", reduction="mean")
    loss = criterion(predicted, actual)

    print("Loss:", loss.item())
