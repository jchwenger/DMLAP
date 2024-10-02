import urllib
import pathlib

import numpy as np

import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# Function to draw landmarks on the image
def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in pose_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    return annotated_image


# Path to the model file
model_path = pathlib.Path("pose_landmarker.task")

# Check if the model file exists, if not, download it
if not model_path.exists():
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    print(f"Downloading model from {url}...")
    urllib.request.urlretrieve(url, model_path)
    print(f"Model downloaded and saved as {model_path}")

# Initialize PoseLandmarker
base_options = python.BaseOptions(model_asset_path="pose_landmarker.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options, output_segmentation_masks=True
)
detector = vision.PoseLandmarker.create_from_options(options)

# Segmentation mask display modes
segmentation_mode = 0  # 0: No mask, 1: Transparent overlay, 2: Only mask

# Open webcam video stream
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip frame horizontally (like a mirror)
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB format (as expected by MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect pose landmarks in the current frame
    detection_result = detector.detect(mp_image)

    # Draw pose landmarks on the frame
    annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)

    # Handle segmentation mask based on the current mode
    if detection_result.segmentation_masks:
        segmentation_mask = detection_result.segmentation_masks[0].numpy_view()

        if segmentation_mode == 1:  # Transparent overlay
            visualized_mask = (
                np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
            )
            visualized_mask = visualized_mask.astype(np.uint8)

            # Blend the segmentation mask with the frame (semi-transparent overlay)
            alpha = 0.5  # Transparency factor
            annotated_frame = cv2.addWeighted(
                annotated_frame, 1 - alpha, visualized_mask, alpha, 0
            )

        elif segmentation_mode == 2:  # Only the segmentation mask
            visualized_mask = (
                np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
            )
            visualized_mask = visualized_mask.astype(np.uint8)
            annotated_frame = visualized_mask

    # Display the annotated frame
    cv2.imshow("Pose Detection", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

    # Capture keypress to toggle segmentation mask display mode
    key = cv2.waitKey(5) & 0xFF
    if key == ord("q"):  # Exit on 'q'
        break
    elif key == ord("1"):  # Toggle segmentation mask display mode
        segmentation_mode = (segmentation_mode + 1) % 3  # Cycles through 0, 1, 2

cap.release()
cv2.destroyAllWindows()
