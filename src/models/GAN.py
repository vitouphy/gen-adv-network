
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from .Generators import *
from .Discriminators import *

class BaseGAN(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.generator = FCGenerator()
        self.discriminator = FCDiscriminator()

    def forward(self, x):
        pass

    def generate(self, z):
        gen_img = self.generator(z).squeeze()
        gen_img = (gen_img + 1) / 2 # renormalize back to [0,1]
        gen_img = gen_img.squeeze().data.cpu()
        return gen_img

    def get_noise(self, x):
        cur_epoch = self.trainer.current_epoch
        noise = torch.randn_like(x)
        noise /= (5*(cur_epoch+1))
        return noise

    def training_step(self, batch, batch_idx, optimizer_idx):

        # discriminator
        if optimizer_idx == 0:
            x, label = batch
            label = torch.ones_like(label).float()
            y_hat_real = self.discriminator(x)
            real_loss = F.binary_cross_entropy(y_hat_real, label)

            z = torch.randn((x.size(0), 100))
            label = torch.zeros_like(label).float()
            y_hat_fake = self.discriminator(self.generator(z))
            fake_loss = F.binary_cross_entropy(y_hat_fake, label)

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
            label = torch.ones_like(label).float()
            z = torch.randn((x.size(0), 100))
            gen_img = self.generator(z)
            gen_img = gen_img + self.get_noise(gen_img)
            y_hat = self.discriminator(gen_img)
            loss = F.binary_cross_entropy(y_hat, label)

            # logs
            tqdm_dict = {'train_g_loss': loss}
            output = {
                'loss': loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            }
            return output

    def validation_step(self, batch, batch_idx):
        output = self.training_step(batch, batch_idx, 1)
        return { 'val_loss': output['loss'] }

    def configure_optimizers(self):
        opt_generator = torch.optim.Adam(self.generator.parameters(), lr=self.args.gen_lr, betas=(0.5, 0.99))
        opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.disc_lr, betas=(0.5, 0.99))
        return [opt_discriminator, opt_generator]


class GAN (BaseGAN):
    def __init__(self, args):
        super().__init__(args)


class WGAN (BaseGAN):
    def __init__(self, args):
        super().__init__(args)

    def training_step(self, batch, batch_idx, optimizer_idx):

        # discriminator
        if optimizer_idx == 0:
            x, _ = batch
            y_hat_real = self.discriminator(x)

            z = torch.randn((x.size(0), 100))
            y_hat_fake = self.discriminator(self.generator(z))

            loss = -(y_hat_real - y_hat_fake).mean()

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
            x, _ = batch
            z = torch.randn((x.size(0), 100))
            gen_img = self.generator(z)
            gen_img = gen_img + self.get_noise(gen_img)
            y_hat = self.discriminator(gen_img)

            loss = -(y_hat).mean()

            # logs
            tqdm_dict = {'train_g_loss': loss}
            output = {
                'loss': loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            }
            return output
