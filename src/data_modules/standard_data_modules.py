import random
import pytorch_lightning as pl
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class StandardDataModule(pl.LightningDataModule):

    def __init__(self, dir_path, batch_size=32):
          super().__init__()
          self.batch_size = batch_size
          self.dir_path = dir_path
          self.image_transforms = transforms.Compose([
              transforms.Resize((32,32)),
              transforms.ToTensor(),
              transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
          ])

    def prepare_data(self):
        data = datasets.ImageFolder(self.dir_path, transform=self.image_transforms)
        N = len(data)
        indices = list(range(N))
        random.shuffle(indices)

        self.train_indices = indices[:int(N*0.7)]
        self.valid_indices = indices[int(N*0.7):int(N*0.85)]
        self.test_indices  = indices[int(N*0.85):]

    def train_dataloader(self):
        data = datasets.ImageFolder(self.dir_path, transform=self.image_transforms)
        train_sampler = SubsetRandomSampler(self.train_indices)
        return DataLoader(data, sampler=train_sampler, batch_size=self.batch_size)

    def val_dataloader(self):
        data = datasets.ImageFolder(self.dir_path, transform=self.image_transforms)
        valid_sampler = SubsetRandomSampler(self.valid_indices)
        return DataLoader(data, sampler=valid_sampler, batch_size=self.batch_size)
