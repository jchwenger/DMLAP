import os
import pathlib

import numpy as np

import torch
from torchvision.transforms import v2

from py5canvas import *

# Get cpu, gpu or mps device for training
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Create video input
w, h = 512, 256

# Empty result initially
result = np.zeros((256, 256, 3))

# model available here:
# https://drive.usercontent.google.com/u/0/uc?id=1OmSp5ymSFHu_M4ZJPmt-u-DROCalFRzK&export=download
PIX2PIX_PATH = (
    "../../python/models/pix2pix_rembrandt/pix2pix_rembrandt.iter_10879_scripted.pt"
)

# Load pix2pix model
G = torch.jit.load(PIX2PIX_PATH, map_location=device)

print(G)
print()
print(f"Our model has {sum(p.numel() for p in G.parameters()):,} parameters.")


def generate(model, image):
    # from (h,w,c) to (c,h,w)
    image = torch.permute(torch.tensor(image.copy()), (2, 0, 1))

    image = v2.ToImage()(image)

    # Check for grayscale (single channel) or with alpha (four) (with GPT 4o)
    if image.shape[0] == 1:
        # Convert grayscale to RGB by repeating the channel 3 times
        image = image.repeat(3, 1, 1)
    elif image.shape[0] == 4:
        # Remove the alpha channel by taking the first 3 channels (RGB)
        image = image[:3, :, :]

    image = v2.Resize((256, 256), antialias=True)(image)
    # from [0,255] to [0,1]
    image = v2.ToDtype(torch.float32, scale=True)(image)
    image = image.to(device)
    # add a batch dimension
    image = image[None, ...]
    with torch.no_grad():
        outputs = model(image).detach().cpu()
    output = outputs[0].permute(1, 2, 0) * 0.5 + 0.5
    return output.numpy()


def brush(pos, delta):
    size = np.linalg.norm(delta)
    fill(255, 50)
    no_stroke()
    circle(pos, np.exp(-size * 0.03) * 10)


def setup():
    create_canvas(w, h)
    frame_rate(20)  # Our framerate will be pretty low anyhow
    background(0)


def draw():
    stroke(255)
    no_fill()
    stroke_weight(1.0)
    if dragging:
        brush(mouse_pos, mouse_delta)

    image(result, [sketch.width / 2, 0], [sketch.width / 2, sketch.width / 2])


def key_pressed(key, modifier):
    print(f"key pressed: {key}")

    # clear
    if key == "c":
        background(0)


def mouse_released():
    global result
    print("generating")
    img = get_image()[:, : sketch.width // 2, :]
    result = generate(G, img)


run()
