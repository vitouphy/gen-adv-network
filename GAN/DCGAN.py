import random
import os
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
#DIR_PATH = "/data/vitou/100DaysofCode/datasets/stanford-dogs-dataset/images/Images"
DIR_PATH = "/data/vitou/100DaysofCode/datasets/mnistasjpg/trainingSet/trainingSet"

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 32, 5, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 5, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),

            nn.Linear(64*6*6, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        h = self.fc(x)
        return h

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc = nn.Sequential(
                nn.Linear(100, 1024*4*4),
                nn.BatchNorm1d(1024*4*4),
                #nn.ReLU(True)
                nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 3,  stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            #nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, 5,  stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            #nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 5,  stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            #nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 1, 5,  stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 1024, 4, 4)
        return self.fc2(x)

class DCGAN(LightningModule):

    def __init__(self):
        super().__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

    def forward(self, x):
        pass

    def prepare_data(self):
        image_transforms = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),
        ])

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
            noise = torch.randn_like(x) * 0.5
            x += noise
            label = torch.ones_like(label).type(torch.FloatTensor).to(device)
            y_hat_real = self.discriminator(x)
            real_loss = F.binary_cross_entropy(y_hat_real.squeeze(), label)
            
            z = torch.randn((x.size(0), 100)).to(device)
            label = torch.zeros_like(label).type(torch.FloatTensor).to(device)
            gen_img = self.generator(z)
            noise = torch.randn_like(gen_img) * 0.5
            gen_img += noise
            y_hat_fake = self.discriminator(gen_img)
            fake_loss = F.binary_cross_entropy(y_hat_fake.squeeze(), label)
            
            loss = real_loss + fake_loss
            
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
            label = torch.ones_like(label).type(torch.FloatTensor).to(device)
            z = torch.randn((x.size(0), 100)).to(device)
            gen_imgs = self.generator(z)
            noise = torch.randn_like(gen_imgs) * 0.5
            gen_imgs = gen_imgs + noise
            y_hat = self.discriminator(gen_imgs)
            loss = F.binary_cross_entropy(y_hat.squeeze(), label)
            
            # logs
            tqdm_dict = {'train_g_loss': loss}
            output = {
                'loss': loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            }
            return output
     
    def training_epoch_end(self, outputs):
        cur_epoch = self.trainer.current_epoch
        if cur_epoch % 25 == 0:
            checkpoint_callback = self.trainer.checkpoint_callback
            filepath = os.path.join(checkpoint_callback.dirpath, "epoch{}.ckpt".format(cur_epoch))
            trainer.save_checkpoint(filepath)
        return { 'test': None } # we do not need to return anything, but cuz of current pl version
    
    def show_current_generation(self):
        z = torch.randn((1,100)).to(device)
        gen_img = self.generator(z).squeeze()
        #score = self.discriminator(gen_img)
        gen_img.unsqueeze_(0)
        gen_img = gen_img.repeat(3, 1, 1)
        gen_img = gen_img.squeeze().data.cpu()
        #img = transforms.ToPILImage()(gen_img).convert("RGB")
        #print ("Score of this Generation: ", score)
        #plt.imshow(img)
        #plt.show()
        #self.experiment.add_image("Generated Image {}, img, self.trainer.global_step)
        return gen_img
    
    def generate(self, z):
#         z = torch.randn((1,100)).to(device)
        gen_img = self.generator(z).squeeze()
        gen_img.unsqueeze_(0)
        gen_img = gen_img.repeat(3, 1, 1)
        gen_img = gen_img.squeeze().data.cpu()
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
        opt_generator = torch.optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.5, 0.99))
        opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.99))
        return [opt_discriminator, opt_generator]
        #return (
        #    {'optimizer': opt_discriminator, 'frequency': 1},
        #    {'optimizer': opt_generator, 'frequency': 1}
        #)

    def train_dataloader(self):
        image_transforms = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((55,55)),
            transforms.ToTensor(),
            #transforms.Normalize(0.5,0.5)
            transforms.Normalize((0.5,),(0.5,))
            #transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])
        data = datasets.ImageFolder(DIR_PATH, transform=image_transforms)
        train_sampler = SubsetRandomSampler(self.train_indices)
        return DataLoader(data, sampler=train_sampler, batch_size=128)
    
    def val_dataloader(self):
        image_transforms = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((55,55)),
            transforms.ToTensor(),
            #transforms.Normalize(0.5,0.5)
            transforms.Normalize((0.5, ),(0.5, ))
            #transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])
        data = datasets.ImageFolder(DIR_PATH, transform=image_transforms)
        valid_sampler = SubsetRandomSampler(self.valid_indices)
        return DataLoader(data, sampler=valid_sampler, batch_size=128)


if __name__ == "__main__":
    model = DCGAN().to(device)
    trainer = pl.Trainer(gpus=1)    
    trainer.fit(model)
