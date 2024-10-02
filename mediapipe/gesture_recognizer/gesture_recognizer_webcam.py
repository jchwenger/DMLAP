import urllib
import pathlib

import numpy as np

import cv2

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python import vision

# Constants
MARGIN = 10  # pixels
FONT_SIZE = 2
FONT_THICKNESS = 2
HANDEDNESS_TEXT_COLOR = (255, 0, 255)  # magenta

# Path to the model file
model_path = pathlib.Path("gesture_recognizer.task")

# Check if the model file exists, if not, download it
if not model_path.exists():
    url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"
    print(f"Downloading model from {url}...")
    urllib.request.urlretrieve(url, model_path)
    print(f"Model downloaded and saved as {model_path}")

# Initialize MediaPipe GestureRecognizer
base_options = base_options_module.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(base_options=base_options, num_hands=2)
recognizer = vision.GestureRecognizer.create_from_options(options)

print("----------------------------------------")
print("Available gestures:")
print(
    "\n".join(
        [
            "None",
            "Closed_Fist ✊",
            "Open_Palm 👋",
            "Pointing_Up ☝️",
            "Thumb_Down 👎",
            "Thumb_Up 👍",
            "Victory  ✌️ ",
            "ILoveYou 🤟",
        ]
    )
)
print("----------------------------------------")


# Function to draw landmarks and recognized gestures on the image
def draw_landmarks_and_gestures_on_image(rgb_image, recognition_result):
    hand_landmarks_list = recognition_result.hand_landmarks
    handedness_list = recognition_result.handedness
    gesture_list = recognition_result.gestures
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands and gestures to visualize
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        corrected_handedness = (
            "Right" if handedness[0].category_name == "Left" else "Left"
        )
        gesture = gesture_list[idx]

        # Draw hand landmarks
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in hand_landmarks
            ]
        )
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style(),
        )

        # Get the top left corner of the detected hand's bounding box
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image
        cv2.putText(
            annotated_image,
            f"{corrected_handedness}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

        # Draw recognized gesture on the image
        if gesture:
            gesture_text = gesture[0].category_name
            cv2.putText(
                annotated_image,
                f"{gesture_text}",
                (text_x, text_y - 70),  # Display gesture above handedness
                cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE,
                HANDEDNESS_TEXT_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

    return annotated_image


# Open webcam video stream
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB and create MediaPipe Image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Perform gesture recognition in the frame
    recognition_result = recognizer.recognize(mp_image)

    # Annotate frame with detected landmarks and gestures
    if recognition_result:
        annotated_frame = draw_landmarks_and_gestures_on_image(
            frame, recognition_result
        )
    else:
        annotated_frame = frame

    # Display the annotated frame
    cv2.imshow("Gesture Recognition", annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
