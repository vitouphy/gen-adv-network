import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#DIR_PATH = "/kaggle/input/stanford-dogs-dataset/images/Images"
DIR_PATH = "/data/vitou/100DaysofCode/datasets/stanford-dogs-dataset/images/Images"
#DIR_PATH = "/data/vitou/100DaysofCode/datasets/mnistasjpg/trainingSet/trainingSet"

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
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
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        h = self.fc(x)
        return h

class Generator(nn.Module):
    def __init__(self, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 3*32*32)
        )
    def forward(self, z):
        img = self.fc(z)
        return img.view(-1, *self.img_shape)


class GAN(LightningModule):

    def __init__(self):
        super().__init__()
        self.generator = Generator((3,32,32))
        self.discriminator = Discriminator()

    def forward(self, x):
        pass

    def prepare_data(self):
        image_transforms = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
        ])

        dir_path = '/kaggle/input/'
        data = datasets.ImageFolder(DIR_PATH, transform=image_transforms)
        
        N = len(data)
        indices = list(range(N))
        random.shuffle(indices)
        
        self.train_indices = indices[:int(N*0.7)]
        self.valid_indices = indices[int(N*0.7):int(N*0.85)]
        self.test_indices  = indices[int(N*0.85):]


    def training_step(self, batch, batch_idx, optimizer_idx):
        
        # discriminator
        if optimizer_idx == 0:
            x, label = batch
            x, label = x.to(device), label.to(device)
            #label = torch.ones_like(label).to(device)
            y_hat_real = self.discriminator(x)
            #real_loss = F.cross_entropy(y_hat_real, label)
            
            z = torch.randn((x.size(0), 128)).to(device)
            #label = torch.zeros_like(label).to(device)
            y_hat_fake = self.discriminator(self.generator(z))
            #fake_loss = F.cross_entropy(y_hat_fake, label)
            
            loss = -(y_hat_real - y_hat_fake).mean()

            #loss = real_loss + fake_loss
            
            # logs
            tqdm_dict = {'train_d_loss': loss}
            output = {
                'loss': loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            }
            return output
        
        # generator
        elif optimizer_idx == 1:
            x, label = batch
            x, label = x.to(device), label.to(device)
            z = torch.randn((x.size(0), 128)).to(device)
            y_hat = self.discriminator(self.generator(z))
            #loss = F.cross_entropy(y_hat, label)
            loss = -y_hat.mean()
            
            # logs
            tqdm_dict = {'train_g_loss': loss}
            output = {
                'loss': loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            }
            return output
     
    
    def show_current_generation(self):
        z = torch.randn((1,128)).to(device)
        gen_img = self.generator(z)
        #score = self.discriminator(gen_img)
        
        gen_img = gen_img.squeeze().data.cpu()
        #img = transforms.ToPILImage()(gen_img).convert("RGB")
        #print ("Score of this Generation: ", score)
        #plt.imshow(img)
        #plt.show()
        #self.experiment.add_image("Generated Image {}, img, self.trainer.global_step)
        return gen_img

    def validation_step(self, batch, batch_idx):
        
        # show the first image
        if batch_idx == 0:
            for i in range(5):
                img = self.show_current_generation()
                epoch = self.trainer.current_epoch #global_step
                step = self.trainer.global_step
                self.logger.experiment.add_image("Generation {}-{}".format(epoch, i), img, step)
        
        output = self.training_step(batch, batch_idx, 1)
        return {
            'val_loss': output['loss']
        }
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}
        
    def configure_optimizers(self):
        opt_generator = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=4e-4)
        return [opt_discriminator, opt_generator]

    def train_dataloader(self):
        image_transforms = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
        ])
        data = datasets.ImageFolder(DIR_PATH, transform=image_transforms)
        train_sampler = SubsetRandomSampler(self.train_indices)
        return DataLoader(data, sampler=train_sampler, batch_size=64)
    
    def val_dataloader(self):
        image_transforms = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
        ])
        data = datasets.ImageFolder(DIR_PATH, transform=image_transforms)
        valid_sampler = SubsetRandomSampler(self.valid_indices)
        return DataLoader(data, sampler=valid_sampler, batch_size=64)

model = GAN().to(device)
trainer = pl.Trainer(gpus=1)    
trainer.fit(model)
