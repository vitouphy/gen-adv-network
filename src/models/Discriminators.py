import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models


class FCDiscriminator(nn.Module):
    def __init__(self):
        super(FCDiscriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
#             nn.MaxPool2d(2, stride=2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(32, 32, 3, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
#             nn.MaxPool2d(2, stride=2),

            nn.Conv2d(32, 64, 5, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Flatten(),

            nn.Linear(1024, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.fc(x)
        return h


class CNNDiscriminator(nn.Module):
    def __init__(self):
        super(CNNDiscriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 32, 5, padding=2, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 5, padding=2, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(64*4*4, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        h = self.fc(x)
        return h
