import os
import pathlib

import cv2
import numpy as np

from py5canvas import *

from numpy.linalg import norm

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F

import torchvision as tv
from torchvision.transforms import v2
import torchvision.transforms.functional as TF

# Get cpu, gpu or mps device for training
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
device = "cpu" # bug with conv_transpose2d on mps
print(f"Using {device} device")

# Output size
w, h = 512, 512
# Make sure this matches the latent dimension you trained on
latent_dim = 100

def random_latent_vector():
    return torch.randn(1,100,1,1) * 1.0 # Try varying this multiplier

# utils
def denorm(x):
    """Denormalize the outputs from [-1, 1] to [0,1] (generator with 'tanh' activation)"""
    return (x * 0.5) + 0.5

seed1 = random_latent_vector()
seed2 = random_latent_vector()

a = seed1
b = seed1 @ seed2

# Number of frames for interpolation (more, slower)
n_frames = 100

# Load generator model
DCGAN_PATH = pathlib.Path("../models/dcgan_fashion_mnist/dcgan_fashion_mnist_g.iter_2339_scripted.pt")
G = torch.jit.load(DCGAN_PATH, map_location=device)
print(G)
print()
print(f"Our model has {sum(p.numel() for p in G.parameters()):,} parameters.")

def slerp(val, low, high):
    # Compute the cosine of the angle between the vectors and clip 
    # it to avoid out-of-bounds errors
    omega = torch.acos(torch.clamp(low/torch.norm(low) @ high / torch.norm(high) -1.0, 1.0))
    so = torch.sin(omega)
    return torch.where(
        so == 0,
        # If sin(omega) is 0, use LERP (linear interpolation)
        (1.0 - val) * low + val * high,
        # Otherwise perform spherical interpolation (SLERP)
        (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high
    )

# Runs the model on an input image
def generate_image(model):
    noise = slerp(((sketch.frame_count) % n_frames)/n_frames, a, b)
    with torch.no_grad():
        generated_image = G(noise).detach().cpu().squeeze().numpy()
    generated_image = np.clip(generated_image, -1, 1)
    return generated_image*0.5 + 0.5

def setup():
    sketch.create_canvas(w, h)
    sketch.frame_rate(60)

def draw():
    c = sketch.canvas

    global a, b # We neeed this to modify a and b

    if sketch.frame_count % n_frames == 0:
        a, b = b, a
        b = random_latent_vector()

    c.background(0)
    img = generate_image(G)
    c.image(img, [0,0], [c.width, c.height], opacity=1)

run()
