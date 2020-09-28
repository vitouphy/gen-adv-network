import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models


class FCGenerator(nn.Module):
    ''' fully conntected generator '''
    ''' given noise of dimension 100, it will generate 32x32 images '''
    def __init__(self):
        super(FCGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 3*32*32)
        )
    def forward(self, z):
        img = self.fc(z)
        return img.view(-1, 3, 32, 32)


class CNNGenerator(nn.Module):
    '''
    use strided convolution and cnn as generator.
    given noise of dimension 100, it will generate 32x32 images
    '''

    def __init__(self):
        super(CNNGenerator, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(100, 512*4*4),
            nn.BatchNorm1d(512*4*4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4,  stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4,  stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 3, 4,  stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 4, 4)
        return self.fc2(x)
