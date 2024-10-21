#!/usr/bin/env python3
import numpy as np
from py5canvas import canvas
from mlxtend.image import extract_face_landmarks
import tensorflow as tf
from tensorflow import keras
import cv2

# Create video input
w, h = 512, 256

model_path = '../models/edges2comics/e100_generator.hd5'
# Load pix2pix model
generator = keras.models.load_model(model_path)
print(generator.summary())

# Runs the model on an input image
def generate_image(model, img):
    img = (img/255)*2.0 - 1.0 # Scale to [-1,1]
    res = model(np.expand_dims(img, 0), training=True)[0].numpy()
    return (res*0.5 + 0.5) # Scale to [0, 1] (Canvas is fine with that)


def setup():
    sketch.create_canvas(w, h)
    sketch.frame_rate(20) # Our framerate will be pretty low anyhow


def draw():
    c = sketch.canvas
    # We draw on black background and white strokes
    c.background(0)
    c.stroke(255)
    c.no_fill()
    c.stroke_weight(1.0)
    # Geneate a simple animation
    np.random.seed(120)
    c.push()
    c.translate(c.height/2, c.height/2) # Translate to center of left part of canvas
    c.rotate(sketch.frame_count*0.1)    # and rotate continuosly
    # draw some random circles
    for i in range(20):
        c.circle(np.random.uniform(-c.height/2, c.height/2, 2), np.random.uniform(5, 30))
    c.pop()

    # Get half of canvas
    img = c.get_image()[:,:256,:]
    # Generate and draw result
    result = generate_image(generator, img)
    c.image(result, [256,0], [256, 256])
