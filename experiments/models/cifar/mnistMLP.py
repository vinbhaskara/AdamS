'''adampapercifar for CIFAR10. FC layers are removed. Paddings are adjusted.
Modified by Vineeth S. Bhaskara and Sneha Desai (Winter 2019).
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['mnistmlp']



class MnistMLP(nn.Module):

    def __init__(self, num_classes=10, dropout=False):
        super(MnistMLP, self).__init__()
        
        self.dropout = dropout
        if self.dropout:
            print('Using Dropout 0.5 for Model.')
        
        self.fc1 = nn.Linear(784, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        
                
        if self.dropout: # just here
            x = F.dropout(x, p=0.5, training=self.training)
        
        
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def mnistmlp(**kwargs):

    model = MnistMLP(**kwargs)
    return model
