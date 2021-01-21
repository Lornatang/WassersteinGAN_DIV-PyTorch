# WassersteinGAN_DIV-PyTorch

### Overview

This repository contains an op-for-op PyTorch reimplementation
of [Wasserstein Divergence for GANs](http://xxx.itp.ac.cn/abs/1712.01026).

### Table of contents

1. [About Wasserstein Divergence for GANs](#about-wasserstein-divergence-for-gans)
2. [Model Description](#model-description)
3. [Installation](#installation)
    * [Clone and install requirements](#clone-and-install-requirements)
    * [Download pretrained weights](#download-pretrained-weights-eg-lsun)
    * [Download cartoon faces](#download-cartoon-faces)
4. [Test](#test)
    * [Torch Hub call](#torch-hub-call)
    * [Base call](#base-call)
5. [Train](#train-eg-lsun)
6. [Contributing](#contributing)
7. [Credit](#credit)

### About Wasserstein Divergence for GANs

If you're new to WassersteinGAN-DIV, here's an abstract straight from the paper:

In many domains of computer vision, generative adversarial networks (GANs) have achieved great success, among which the
fam- ily of Wasserstein GANs (WGANs) is considered to be state-of-the-art due to the theoretical contributions and
competitive qualitative performance. However, it is very challenging to approximate the k-Lipschitz constraint required
by the Wasserstein-1 metric (W-met). In this paper, we propose a novel Wasserstein divergence (W-div), which is a
relaxed version of W-met and does not require the k-Lipschitz constraint.As a concrete application, we introduce a
Wasserstein divergence objective for GANs (WGAN-div), which can faithfully approximate W-div through optimization. Under
various settings, including progressive growing training, we demonstrate the stability of the proposed WGAN-div owing to
its theoretical and practical advantages over WGANs. Also, we study the quantitative and visual performance of WGAN-div
on standard image synthesis benchmarks, showing the superior performance of WGAN-div compared to the state-of-the-art
methods.

### Model Description

We have two networks, G (Generator) and D (Discriminator).The Generator is a network for generating images. It receives
a random noise z and generates images from this noise, which is called G(z).Discriminator is a discriminant network that
discriminates whether an image is real. The input is x, x is a picture, and the output is D of x is the probability that
x is a real picture, and if it's 1, it's 100% real, and if it's 0, it's not real.

### Installation

#### Clone and install requirements

```shell
$ git clone https://github.com/Lornatang/WassersteinGAN_DIV-PyTorch.git
$ cd WassersteinGAN_DIV-PyTorch/
$ pip3 install -r requirements.txt
```

#### Download pretrained weights (e.g. LSUN)

```shell
$ cd weights/
$ python3 download_weights.py
```

#### Download cartoon faces

[baiduclouddisk](https://pan.baidu.com/s/1nawrN1Kiw3Z2Jk1NgJqZTQ)  access: `68rn`

### Test

#### Torch hub call

```python
# Using Torch Hub library.
import torch
import torchvision.utils as vutils

# Choose to use the device.
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load the model into the specified device.
model = torch.hub.load("Lornatang/WassersteinGAN_DIV-PyTorch", "lsun", pretrained=True, progress=True, verbose=False)
model.eval()
model = model.to(device)

# Create random noise image.
num_images = 64
noise = torch.randn(num_images, 100, 1, 1, device=device)

# The noise is input into the generator model to generate the image.
with torch.no_grad():
    generated_images = model(noise)

# Save generate image.
vutils.save_image(generated_images, "lsun.png", normalize=True)
```

#### Base call

Using pre training model to generate pictures.

```text
usage: test.py [-h] [-a ARCH] [-n NUM_IMAGES] [--outf PATH] [--device DEVICE]

An implementation of WassersteinGAN-DIV algorithm using PyTorch framework.

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture: _gan | discriminator |
                        load_state_dict_from_url | lsun (default: cifar10)
  -n NUM_IMAGES, --num-images NUM_IMAGES
                        How many samples are generated at one time. (default:
                        64).
  --outf PATH           The location of the image in the evaluation process.
                        (default: ``test``).
  --device DEVICE       device id i.e. `0` or `0,1` or `cpu`. (default:
                        ``cpu``).

# Example (e.g. LSUN)
$ python3 test.py -a lsun --device cpu
```

<span align="center"><img src="assets/mnist.gif" alt="">
</span>

### Train (e.g. LSUN)

```text
usage: train.py [-h] --dataset DATASET [-a ARCH] [-j N] [--start-iter N]
                [--iters N] [-b N] [--lr LR] [--n_critic N_CRITIC]
                [--clip_value CLIP_VALUE] [--image-size IMAGE_SIZE]
                [--classes CLASSES] [--pretrained] [--netD PATH] [--netG PATH]
                [--manualSeed MANUALSEED] [--device DEVICE]
                DIR

An implementation of WassersteinGAN-DIV algorithm using PyTorch framework.

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     | lsun |.
  -a ARCH, --arch ARCH  model architecture: _gan | discriminator |
                        load_state_dict_from_url | lsun (default: lsun)
  -j N, --workers N     Number of data loading workers. (default:8)
  --start-iter N        manual iter number (useful on restarts)
  --iters N             The number of iterations is needed in the training of
                        model. (default: 500000)
  -b N, --batch-size N  mini-batch size (default: 64), this is the total batch
                        size of all GPUs on the current node when using Data
                        Parallel or Distributed Data Parallel.
  --lr LR               Learning rate. (default:0.00005)
  --n_critic N_CRITIC   Number of training steps for discriminator per iter.
                        (Default: 5).
  --clip_value CLIP_VALUE
                        Lower and upper clip value for disc. weights.
                        (Default: 0.01).
  --image-size IMAGE_SIZE
                        The height / width of the input image to network.
                        (default: 64).
  --classes CLASSES     comma separated list of classes for the lsun data set.
                        (default: ``church_outdoor``).
  --pretrained          Use pre-trained model.
  --netD PATH           Path to latest discriminator checkpoint. (default:
                        ````).
  --netG PATH           Path to latest generator checkpoint. (default: ````).
  --manualSeed MANUALSEED
                        Seed for initializing training. (default:1111)
  --device DEVICE       device id i.e. `0` or `0,1` or `cpu`. (default:
                        ``0``).

# Example (e.g. LSUN)
$ python3 train.py data -a lsun --dataset lsun --image-size 64 --classes church_outdoor --pretrained --device 0
```

If you want to load weights that you've trained before, run the following command.

```bash
$ python3 train.py data \
                   -a lsun \
                   --dataset lsun \
                   --image-size 64 \
                   --classes church_outdoor \
                   --start-iter 10000 \
                   --netG weights/lsun_G_iter_10000.pth \
                   --netD weights/lsun_D_iter_10000.pth
```

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### Wasserstein Divergence for GANs

*Jiqing Wu, Zhiwu Huang, Janine Thoma, Dinesh Acharya, Luc Van Gool*

**Abstract** <br>
In many domains of computer vision, generative adversarial networks (GANs) have achieved great success, among which the
family of Wasserstein GANs (WGANs) is considered to be state-of-the-art due to the theoretical contributions and
competitive qualitative performance. However, it is very challenging to approximate the k-Lipschitz constraint required
by the Wasserstein-1 metric~(W-met). In this paper, we propose a novel Wasserstein divergence~(W-div), which is a
relaxed version of W-met and does not require the k-Lipschitz constraint. As a concrete application, we introduce a
Wasserstein divergence objective for GANs~(WGAN-div), which can faithfully approximate W-div through optimization. Under
various settings, including progressive growing training, we demonstrate the stability of the proposed WGAN-div owing to
its theoretical and practical advantages over WGANs. Also, we study the quantitative and visual performance of WGAN-div
on standard image synthesis benchmarks of computer vision, showing the superior performance of WGAN-div compared to the
state-of-the-art methods.

[[Paper]](http://xxx.itp.ac.cn/pdf/1712.01026)

```
@article{DBLP:journals/corr/abs-1712-01026,
  author    = {Jiqing Wu and
               Zhiwu Huang and
               Janine Thoma and
               Luc Van Gool},
  title     = {Energy-relaxed Wasserstein GANs(EnergyWGAN): Towards More Stable and
               High Resolution Image Generation},
  journal   = {CoRR},
  volume    = {abs/1712.01026},
  year      = {2017},
  url       = {http://arxiv.org/abs/1712.01026},
  archivePrefix = {arXiv},
  eprint    = {1712.01026},
  timestamp = {Mon, 13 Aug 2018 16:48:51 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1712-01026.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
