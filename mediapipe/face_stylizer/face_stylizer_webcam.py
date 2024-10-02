import urllib
import pathlib

import numpy as np

import cv2

import mediapipe as mp
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python import vision

# Height and width that will be used by the model
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

# Function to resize and crop the frame to a fixed size
def resize_frame(image, target_height=256, target_width=256):
    h, w = image.shape[:2]

    # Scale the image while maintaining aspect ratio
    if h < w:
        scale = target_height / h
        new_w = int(w * scale)
        resized_image = cv2.resize(image, (new_w, target_height))
    else:
        scale = target_width / w
        new_h = int(h * scale)
        resized_image = cv2.resize(image, (target_width, new_h))

    # Now crop to the desired width and height
    start_x = (resized_image.shape[1] - target_width) // 2
    start_y = (resized_image.shape[0] - target_height) // 2

    cropped_image = resized_image[
        start_y : start_y + target_height, start_x : start_x + target_width
    ]

    return cropped_image


# Initialize model file paths
model_files = {
    "1": pathlib.Path("face_stylizer_color_ink.task"),
    "2": pathlib.Path("face_stylizer_color_sketch.task"),
    "3": pathlib.Path("face_stylizer_oil_painting.task"),
}

for model_path in model_files.values():
    # Check if the model file exists, if not, download it
    if not model_path.exists():
        url = f"https://storage.googleapis.com/mediapipe-models/face_stylizer/blaze_face_stylizer/float32/latest/{model_path}"
        print()
        print(f"Downloading model from {url}...")
        urllib.request.urlretrieve(url, model_path)
        print(f"Model downloaded and saved as {model_path}")


# Function to load the FaceStylizer model
def load_model(model_path):
    base_options = base_options_module.BaseOptions(model_asset_path=model_path)
    options = vision.FaceStylizerOptions(base_options=base_options)
    return vision.FaceStylizer.create_from_options(options)


# Load the initial model (default is model 1)
stylizer = load_model(model_files["1"])

# Open webcam video stream
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally for a mirror-like effect
    frame = cv2.flip(frame, 1)

    # Resize the frame to the desired dimensions
    resized_frame = resize_frame(frame)

    # Convert the frame to RGB format (as expected by MediaPipe)
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe Image from the frame
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Perform face stylization
    stylized_result = stylizer.stylize(mp_image)

    # Check if the stylized result is not None before processing
    if stylized_result:
        # Convert the stylized image back to BGR format for OpenCV
        stylized_frame = stylized_result.numpy_view()
        bgr_stylized_frame = cv2.cvtColor(stylized_frame, cv2.COLOR_RGB2BGR)
    else:
        # If no stylization result, resize the original frame to fixed dimensions
        bgr_stylized_frame = resize_frame(frame)

    # print(bgr_stylized_frame.shape)

    # Display the frame with a fixed size
    cv2.imshow("Stylized Face", bgr_stylized_frame)

    # Capture keypress to change models or exit
    key = cv2.waitKey(5) & 0xFF
    if key == ord("q"):  # Exit on 'q'
        break
    elif key == ord("1"):  # Load model 1 (Color Ink)
        print()
        print(f"Pressed 1, switching to model: {model_files['1']}")
        print()
        stylizer = load_model(model_files["1"])
    elif key == ord("2"):  # Load model 2 (Color Sketch)
        print()
        print(f"Pressed 2, switching to model: {model_files['2']}")
        print()
        stylizer = load_model(model_files["2"])
    elif key == ord("3"):  # Load model 3 (Oil Painting)
        print()
        print(f"Pressed 3, switching to model: {model_files['3']}")
        print()
        stylizer = load_model(model_files["3"])

cap.release()
cv2.destroyAllWindows()
