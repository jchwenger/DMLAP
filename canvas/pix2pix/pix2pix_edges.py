import os
import pathlib

import numpy as np

import cv2
from skimage import feature

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

# models available here:
# https://drive.google.com/file/d/15j05wWFJsMd2OS7sxlFmG4z2exmIzJAK/view?usp=sharing
# https://drive.google.com/file/d/1BScyTUQI3F0DRm5ICGBejNzTsPGWW7O1/view?usp=sharing
PIX2PIX_PATH = (
    "../../python/models/pix2pix_edge2comics/pix2pix_edge2comics.iter_16583_scripted.pt"
    # "../../python/models/pix2pix_edge2comics/edge2comics_60_generator_scripted.pt"
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


# Applies canny edge detection to our input
def apply_canny_skimage(img, sigma=1.5, invert=False):
    """Apply the Scikit-Image Canny edge detector to an image"""
    grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = (feature.canny(grey_img, sigma=sigma) * 255).astype(np.uint8)
    if invert:
        edges = cv2.bitwise_not(edges)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


def apply_canny_cv2(img, thresh1=160, thresh2=250, invert=False):
    """Apply the OpenCV Canny edge detector to an image"""
    grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(grey_img, thresh1, thresh2)
    if invert:
        edges = cv2.bitwise_not(edges)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


canny_fn = [apply_canny_cv2, apply_canny_skimage]
canny_index = 0
apply_canny = canny_fn[canny_index]


def setup():
    create_canvas(w, h)
    frame_rate(20)  # Our framerate will be pretty low anyhow


def draw():
    background(0)

    # Read video and compute landmarks
    frame = vid.read()
    edges = apply_canny(frame)
    res = generate(G, edges)
    if full:
        image(res, [0, 0], [width, height])
    else:
        image(edges, [0, 0], [width // 2, height])
        image(res, [width // 2, 0], [width // 2, height])


def key_pressed(key, modifier):
    global full
    global canny_index
    global apply_canny

    # change canny function with space
    if key == " ":
        canny_index = (canny_index + 1) % 2
        apply_canny = canny_fn[canny_index]

    if key == "f":
        full = not full


run()
