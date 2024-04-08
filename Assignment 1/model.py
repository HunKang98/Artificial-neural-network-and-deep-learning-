
import torch.nn as nn
import torch

class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self, dropout):

        super(LeNet5, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1), nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        if dropout:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=16*25, out_features=120), nn.ReLU(),
                nn.Linear(in_features=120, out_features=84), nn.ReLU(),
                nn.Linear(in_features=84, out_features=10)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=16*25, out_features=120), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(in_features=120, out_features=84), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(in_features=84, out_features=10)
            )


    def forward(self, img):

        output = self.classifier(self.feature(img))

        return output


class CustomMLP(nn.Module):
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """

    def __init__(self):

        super(CustomMLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=60), nn.ReLU(),
            nn.Linear(in_features=60, out_features=10)
        )

    def forward(self, img):

        output = self.classifier(img)

        return output


