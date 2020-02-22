# WassersteinGAN_DIV-PyTorch

### Update (Feb 22, 2020)

The mnist and fmnist models are now available. Their usage is identical to the other models: 
```python
from wgandiv_pytorch import Generator
model = Generator.from_pretrained('g-mnist') 
```

### Overview
This repository contains an op-for-op PyTorch reimplementation of [Wasserstein Divergence for GANs](http://xxx.itp.ac.cn/pdf/1712.01026).

The goal of this implementation is to be simple, highly extensible, and easy to integrate into your own projects. This implementation is a work in progress -- new features are currently being implemented.  

At the moment, you can easily:  
 * Load pretrained Generate models 
 * Use Generate models for extended dataset

_Upcoming features_: In the next few days, you will be able to:
 * Quickly finetune an Generate on your own dataset
 * Export Generate models for production

### Table of contents
1. [About Wasserstein GAN DIV](#about-wasserstein-gan-div)
2. [Model Description](#model-description)
3. [Installation](#installation)
4. [Usage](#usage)
    * [Load pretrained models](#loading-pretrained-models)
    * [Example: Extended dataset](#example-extended-dataset)
    * [Example: Visual](#example-visual)
5. [Contributing](#contributing) 

### About Wasserstein GAN DIV

If you're new to Wasserstein GAN DIV, here's an abstract straight from the paper:

In many domains of computer vision, generative adversarial networks (GANs) have achieved great success, among which the fam- ily of Wasserstein GANs (WGANs) is considered to be state-of-the-art due to the theoretical contributions and competitive qualitative performance. However, it is very challenging to approximate the k-Lipschitz constraint required by the Wasserstein-1 metric (W-met). In this paper, we propose a novel Wasserstein divergence (W-div), which is a relaxed version of W-met and does not require the k-Lipschitz constraint.As a concrete application, we introduce a Wasserstein divergence objective for GANs (WGAN-div), which can faithfully approximate W-div through optimization. Under various settings, including progressive growing training, we demonstrate the stability of the proposed WGAN-div owing to its theoretical and practical advantages over WGANs. Also, we study the quantitative and visual performance of WGAN-div on standard image synthesis benchmarks, showing the superior performance of WGAN-div compared to the state-of-the-art methods.
### Model Description

We have two networks, G (Generator) and D (Discriminator).The Generator is a network for generating images. It receives a random noise z and generates images from this noise, which is called G(z).Discriminator is a discriminant network that discriminates whether an image is real. The input is x, x is a picture, and the output is D of x is the probability that x is a real picture, and if it's 1, it's 100% real, and if it's 0, it's not real.

### Installation

Install from pypi:
```bash
pip install wgandiv_pytorch
```

Install from source:
```bash
git clone https://github.com/Lornatang/WassersteinGAN_DIV-PyTorch.git
cd WassersteinGAN_DIV-PyTorch
pip install -e .
``` 

### Usage

#### Loading pretrained models

Load an Wasserstein GAN DIV:
```python
from wgandiv_pytorch import Generator
model = Generator.from_name("g-mnist")
```

Load a pretrained Wasserstein GAN DIV:
```python
from wgandiv_pytorch import Generator
model = Generator.from_pretrained("g-mnist")
```

#### Example: Extended dataset

As mentioned in the example, if you load the pre-trained weights of the MNIST dataset, it will create a new `imgs` directory and generate 64 random images in the `imgs` directory.

```python
import os
import torch
import torchvision.utils as vutils
from wgandiv_pytorch import Generator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Generator.from_pretrained("g-mnist")
model.to(device)
# switch to evaluate mode
model.eval()

try:
    os.makedirs("./imgs")
except OSError:
    pass

with torch.no_grad():
    for i in range(64):
        noise = torch.randn(64, 100, device=device)
        fake = model(noise)
        vutils.save_image(fake.detach(), f"./imgs/fake_{i:04d}.png", normalize=True)
    print("The fake image has been generated!")
```

#### Example: Visual

```text
cd $REPO$/framework
sh start.sh
```

Then open the browser and type in the browser address [http://127.0.0.1:10004/](http://127.0.0.1:10004/).
Enjoy it.

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.   

I look forward to seeing what the community does with these models! 