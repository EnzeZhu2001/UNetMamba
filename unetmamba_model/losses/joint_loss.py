from torch import nn
from torch.nn.modules.loss import _Loss

__all__ = ["JointLoss", "WeightedLoss"]


class WeightedLoss(_Loss):
    """Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    """

    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, *input):
        return self.loss(*input) * self.weight


class JointLoss(_Loss):
    """
    Wrap two loss functions into one. This class computes a weighted sum of two losses.
    """

    def __init__(self, first: nn.Module, second: nn.Module,
                 # third: nn.Module, fourth: nn.Module,
                 # fifth: nn.Module, sixth: nn.Module,
                 weight=(1.0, 1.0)):
        super().__init__()
        self.first = WeightedLoss(first, weight[0])
        self.second = WeightedLoss(second, weight[1])
        # self.third = WeightedLoss(third, weight(2))
        # self.fourth = WeightedLoss(fourth, weight(3))
        # self.fifth = WeightedLoss(fifth, weight(4))
        # self.sixth = WeightedLoss(sixth, weight(5))

    def forward(self, *input):
        return self.first(*input) + self.second(*input)
            # + self.third(*input) + self.fourth(*input) + self.fifth(*input) + self.sixth(*input)
