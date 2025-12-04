'''adampapercifar for CIFAR10. FC layers are removed. Paddings are adjusted.
Modified by Vineeth S. Bhaskara and Sneha Desai (Winter 2019).
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['mnistlr']


class MnistLR(nn.Module):

    def __init__(self, num_classes=10):
        super(MnistLR, self).__init__()
        self.linear = nn.Linear(784, num_classes)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.linear(x)

        return x


def mnistlr(**kwargs):

    model = MnistLR(**kwargs)
    return model
