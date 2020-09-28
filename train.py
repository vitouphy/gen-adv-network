
import argparse
import pytorch_lightning as pl
from src.models.GAN import *
from src.data_modules.standard_data_modules import StandardDataModule


def init_args():
    """
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-dt', '--dataset', type=str,
                        help='path to the datasets')

    parser.add_argument('-g', '--generator', type=str, default="cnn",
                        help='type of generator, {cnn, fc}')

    parser.add_argument('-d', '--discriminator', type=str, default="cnn",
                        help='type of discriminator, {cnn, fc}')

    parser.add_argument('-l', '--loss', type=str, default="minmax",
                        help='type of generator, {minmax, wasserstein}')

    parser.add_argument('-gen_lr', '--gen_lr', type=float, default=1e-4,
                        help='learning rate of generator')

    parser.add_argument('-disc_lr', '--disc_lr', type=float, default=1e-4,
                        help='learning rate of discriminator')

    return parser.parse_args()

def train():
    # prepare
    args = init_args()
    dm = StandardDataModule(args.dataset)

    if args.loss == "wasserstein":
        model = WGAN(args)
    else:
        model = GAN(args)

    # train
    trainer = pl.Trainer()
    trainer.fit(model, dm)

    # test
    # trainer.test(datamodule=dm)

if __name__ == "__main__":
    train()
