#!/usr/bin/env python3
import numpy as np
from py5canvas import canvas
from mlxtend.image import extract_face_landmarks
import tensorflow as tf
from tensorflow import keras
import cv2

# Create video input
w, h = 512, 512
vid = canvas.VideoInput(size=(w, h))

model_path = '../models/edges2comics/e100_generator.hd5'
# Load pix2pix model
generator = keras.models.load_model(model_path)
print(generator.summary())

# Runs the model on an input image
def generate_image(model, img):
    img = (img/255)*2.0 - 1.0 # Scale to [-1,1]
    res = model(np.expand_dims(img, 0), training=True)[0].numpy()
    return (res*0.5 + 0.5) # Scale to [0, 1] (Canvas is fine with that)

# Applies canny edge detection to our input
def apply_canny_skimage(img, sigma=0.5):
    import cv2
    from skimage import feature
    grayimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = (feature.canny(grayimg, sigma=sigma)*255).astype(np.uint8)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

def setup():
    sketch.create_canvas(w, h)
    sketch.frame_rate(20) # Our framerate will be pretty low anyhow


def draw():
    c = sketch.canvas

    c.background(0)

    # Read video and compute landmarks
    frame = vid.read()
    edges = apply_canny_skimage(frame)
    res = generate_image(generator, cv2.resize(edges, (256, 256)))
    c.image(res, [0,0], [c.width, c.height])
