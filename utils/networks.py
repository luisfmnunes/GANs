from torch import nn
import torch.nn.functional as F

class DiscriminatorNet(nn.Module):
    """
    A three hidden-layer discriminitive neural network (shallow)
    """

    def __init__(self):
        super().__init__()
        n_features = 784
        n_out = 1

        self.fc1 = nn.Linear(n_features, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, n_out)

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        # Hidden 0
        x = self.fc1(x)
        x = self.leakyReLU(x)
        x = self.dropout(x)

        # Hidden 1
        x = self.fc2(x)
        x = self.leakyReLU(x)
        x = self.dropout(x)

        # Hidden 2
        x = self.fc3(x)
        x = self.leakyReLU(x)
        x = self.dropout(x)

        # Output Layer
        x = self.sigmoid(self.fc4(x))

        return x


class GeneratorNet(nn.Module):
    """
    A three hidden-layer generative neural network (shallow)
    """
    def __init__(self):
        super().__init__()
        n_features = 100
        n_out = 784

        self.fc1 = nn.Linear(n_features, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, n_out)

        self.leakyReLU = nn.LeakyReLU(0.2)

        self.tanh = nn.Tanh()

    def forward(self, x):

        # Hidden 0
        x = self.fc1(x)
        x = self.leakyReLU(x)

        # Hidden 1
        x = self.fc2(x)
        x = self.leakyReLU(x)

        # Hidden 2
        x = self.fc3(x)
        x = self.leakyReLU(x)

        # Output Layer
        x = self.tanh(self.fc4(x))

        return x