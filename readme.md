<!-- #GAN Image
![Dogs image after 200 epochs](./results/training_after_200_epochs.png)

#DC-GAN on CIFAR10
Generation after 100 epochs.
![Cifar10 after 100 epochs](./results/cifar10_dcgan.png) -->

## Generative Adversarial Network (GAN) Library

This repository holds a variety of GAN for generating images.

<!-- [Explanation of what GAN is] -->


## Quick Start

#### Installation
```
pip install -r requirements.txt
```

#### Training

Basic training command:
```
python train.py \
--dataset {path_to_dataset} \
--generator cnn \
--discriminator cnn \
--loss minmax
```

More configuration:
```
python train.py \
--dataset {name_of_dataset} \
--generator cnn \
--discriminator fc \
--loss minmax \
--gen_lr 1e-4 \
--disc_lr 1e-4
```

#### Inference

<!-- For generating images:
```
python generate.py -n 5 --model_output {path_to_model}
``` -->

#### To Do:
- Generating interpolation GIF (Coming Soon)
