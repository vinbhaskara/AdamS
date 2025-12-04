'''adampapercifar for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
Modified by Vineeth S. Bhaskara and Sneha Desai (Winter 2019).
'''
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['cifar10cnn']


class AdamCIFAR10(nn.Module):

    def __init__(self, num_classes=10, dropout=False):
        super(AdamCIFAR10, self).__init__()
        self.dropout = dropout
        if self.dropout:
            print('Using Dropout 0.5 for Model.')
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        if num_classes == 10: # CIFAR 10
            self.preclassifier = nn.Sequential(nn.Linear(1152, 1000), nn.ReLU(inplace=True))
            self.classifier = nn.Linear(1000, num_classes)
        elif num_classes == 100: # CIFAR 100 - Trying to keep the total parameters almost SAME
            print('DOING CIFAR 100 MODEL')
            self.preclassifier = nn.Sequential(nn.Linear(1152, 928), nn.ReLU(inplace=True))
            self.classifier = nn.Linear(928, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        if self.dropout:
            x = F.dropout(x, p=0.5, training=self.training)
            
        x = self.preclassifier(x)
        x = self.classifier(x)
        return x


def cifar10cnn(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AdamCIFAR10(**kwargs)
    return model
