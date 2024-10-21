import os
import pathlib

import numpy as np

import matplotlib

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

full = False
w, h = 512, 512
vid = VideoInput(size=(w, h))

# model available here:
# https://drive.google.com/file/d/1H95702mO-7dkYjaJa_Bz4AqUvPL2B7vS/view?usp=sharing
PIX2PIX_PATH = pathlib.Path(
    "../../python/models/pix2pix_facades/pix2pix_facades.iter_8000_scripted.pt"
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


def jet_color(v):
    c = matplotlib.colormaps.get_cmap("jet")
    return (
        np.array(c(v)) * 255
    )  # The output of this function is between 0 and 1, we will use 0 to 255 colors


# this gives us values between 0 and 1 for the labels
labels = np.linspace(0, 1, 12)


def random_label():
    return np.random.choice(labels[2:])  # simply excludes background and facade


# # Draw the main facade
# pad = 0
# fill(jet_color(labels[1]))
# rect(pad, pad, 256-pad*2, 256)

## Draw some random rectangle with random feature colors
# for i in range(30):
#    fill(jet_color(random_label()))
#    rect(np.random.uniform(pad, height-pad*2, size=2), np.random.uniform(2, 7, size=2)*6)
#    #fill(jet_color(random_label()))
#    #circle(np.random.uniform(pad, height-pad, size=2), np.random.uniform(5, height*0.15)*0.5) #, size=2))
## Get the left half of the canvas image
# img = get_image()[:, :256]

# # And transform it using our pix2pix model
# result = generate(model, img.copy())
# image(result, [256, 0])


def setup():
    sketch.create_canvas(w, h)
    sketch.frame_rate(20)  # Our framerate will be pretty low anyhow


def draw():
    # Fill with the darkest color (background)
    background(jet_color(0)[:-1])
    no_stroke()

    push()
    # Translate to center of left part of canvas
    translate(height / 2, height / 2)
    # and rotate continuosly
    rotate(frame_count * 0.1)
    # draw some random circles
    for i in range(10):
        fill(jet_color(random_label()))
        rect(
            np.random.uniform(-height / 2, height / 2, size=2),
            np.random.uniform(5, 30, size=2),
        )
    pop()

    if full:
        # Get half of canvas
        img = get_image()[:, :256, :]

        # Generate and draw result
        result = generate(G, img)
        image(result, [0, 0], [512, 512])
    else:
        # Get half of canvas
        img = get_image()[:, :256, :]

        # Generate and draw result
        result = generate(G, img)
        image(result, [256, 0], [256, 512])


def key_pressed(key, modifier):
    global full
    if key == "f":
        full = not full


run()
