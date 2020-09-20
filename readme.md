<!-- #GAN Image
![Dogs image after 200 epochs](./results/training_after_200_epochs.png)

#DC-GAN on CIFAR10
Generation after 100 epochs.
![Cifar10 after 100 epochs](./results/cifar10_dcgan.png) -->

## Generative Adversarial Network (GAN) Library

This repository holds a variety of GAN for generating images.




## Quick Start

#### Installation
```
pip install -r requirements.txt
```

#### Dataset 
Copy your desired images that GAN should learn to generate in `datasets/{name_of_dataset}` folders.

#### Training

Basic training command:
```
python run.py \
--dataset {name_of_dataset} \
--generator cnn \
--discriminator cnn \
--loss minimax \
--model_output {path_to_model}
```

More configuration:
```
python run.py \
--dataset {name_of_dataset} \
--generator cnn \
--discriminator dense \
--loss minimax \
--model_output {path_to_model}
--gen_lr 1e-4 \
--disc_lr 1e-4 \
--gen_freq 1 \
--disc_freq 1 
```

#### Inference

For generating images:
```
python generate.py -n 5 --model_output {path_to_model}
```

#### To Do:
- Generating interpolation GIF (Coming Soon)

