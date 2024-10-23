import os
import urllib
import pathlib

from PIL import Image

import torch

from py5canvas import *


# Get cpu, gpu or mps device for training
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
device = "cpu"  # bug with conv_transpose2d on mps
print(f"Using {device} device")

# Output size
w, h = 512, 512

# Make sure this matches the latent dimension you trained on
latent_dim = 100


def random_latent_vector():
    return torch.randn(1, 100, 1, 1) * 1.0  # Try varying this multiplier


seed1 = random_latent_vector()
seed2 = random_latent_vector()

a = seed1
b = seed1 @ seed2

# Number of frames for interpolation (more, slower)
n_frames = 100

# Local path to your trained net
# DCGAN_PATH  = pathlib.Path("../../python/models/dcgan_mnist/dcgan_mnist_g.iter_0936_scripted.pt")

# models available here:
# MNIST: https://drive.usercontent.google.com/u/0/uc?id=1ZuePaVjXaAkQJTdeu060ELip6Ri6VMl2&export=download
# FashionMNIST: https://drive.usercontent.google.com/u/0/uc?id=1ZOozan_vi6yCC6ulFtHHUS2jbT_BUUqd&export=download
# CelebA (requires three channels): https://drive.google.com/file/d/1WBfUkjOtgTX2c2UgFYZY4skjMzDaGi2y/view?usp=sharing

DCGAN_PATH = pathlib.Path(
    "../../python/models/dcgan_mnist/dcgan_mnist_g.iter_0936_scripted.pt"
    # "../../python/models/dcgan_fashion_mnist/dcgan_fashion_mnist_g.iter_2339_scripted.pt"
    # "../../python/models/dcgan_fashion_mnist_redux/dcgan_mnist_g.iter_2340_scripted.pt"
    # "../../python/models/dcgan_celeba/dcgan_celeba_g.iter_6791_scripted.pt"
)

# Load generator model
G = torch.jit.load(DCGAN_PATH, map_location=device)

print(G)
print()
print(f"Our model has {sum(p.numel() for p in G.parameters()):,} parameters.")


def slerp(val, low, high):
    # Compute the cosine of the angle between the vectors and clip
    # it to avoid out-of-bounds errors
    omega = torch.acos(
        torch.clamp(low / torch.norm(low) @ high / torch.norm(high) - 1.0, 1.0)
    )
    so = torch.sin(omega)
    return torch.where(
        so == 0,
        # If sin(omega) is 0, use LERP (linear interpolation)
        (1.0 - val) * low + val * high,
        # Otherwise perform spherical interpolation (SLERP)
        (torch.sin((1.0 - val) * omega) / so) * low
        + (torch.sin(val * omega) / so) * high,
    )


# Runs the model on an input image
def generate(model):
    with torch.no_grad():
        noise = slerp(((sketch.frame_count) % n_frames) / n_frames, a, b)
        img = G(noise).detach()[0].cpu()
        img = torch.permute(img, (1, 2, 0))
        img = torch.clip(img, -1, 1)
    return img * 0.5 + 0.5


def setup():
    create_canvas(w, h)
    frame_rate(60)


def draw():
    global a, b  # We neeed this to modify a and b

    if sketch.frame_count % n_frames == 0:
        a, b = b, a
        b = random_latent_vector()

    background(0)
    img = generate(G)
    image(img, [0, 0], [width, height], opacity=1)


run()
