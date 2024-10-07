import time
import urllib
import pathlib
import argparse

import numpy as np

import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Constants for drawing
MARGIN = 30  # pixels
ROW_SIZE = 30  # pixels
FONT_SIZE = 3
FONT_THICKNESS = 2
RECT_COLOR = (255, 0, 0)  # blue (BGR)
TEXT_COLOR = (255, 255, 255)  # white
FPS_AVG_FRAME_COUNT = 10

def visualize(image, detection_result) -> np.ndarray:
    """Draws bounding boxes on the input image and return it.
    Args:
        image: The input RGB image.
        detection_result: The list of all "Detection" entities to be visualized.
    Returns:
        Image with bounding boxes.
    """
    for detection in detection_result.detections:

        # Draw bounding box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, RECT_COLOR, 3)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + " (" + str(probability) + ")"
        # print(result_text)
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(
            image,
            result_text,
            text_location,
            cv2.FONT_HERSHEY_PLAIN,
            FONT_SIZE,
            TEXT_COLOR,
            FONT_THICKNESS,
        )

    return image


def download(url, model_path):
    print()
    print(f"Downloading model from {url}...")
    urllib.request.urlretrieve(url, model_path)
    print(f"Model downloaded and saved as {model_path}")


def show_fps(
    current_frame,
    fps,
    row_size = 40,  # pixels
    left_margin = 24,  # pixels
    text_color = (20, 60, 220),  # crimson (BGR)
    font_size = 2,
    font_thickness = 2,
):
    # Show the FPS
    fps_text = "FPS = {:.1f}".format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(
        current_frame,
        fps_text,
        text_location,
        cv2.FONT_HERSHEY_PLAIN,
        font_size,
        text_color,
        font_thickness,
    )

def run(args):
    """Continuously run inference on images acquired from the camera.
    Args:
      args.model: Name of the TFLite object detection model.
      args.camera_id: The camera id to be passed to OpenCV.
      args.frame_width: The width of the frame captured from the camera.
      args.frame_height: The height of the frame captured from the camera.
    """

    # Check if the model file exists, if not, download it
    if not args.model.exists():
        download(args.url, args.model)

    detection_result_list = []

    def visualize_callback(
        result: vision.ObjectDetectorResult, output_image: mp.Image, timestamp_ms: int
    ):
        result.timestamp_ms = timestamp_ms
        detection_result_list.append(result)

    # Initialize ObjectDetector
    base_options = python.BaseOptions(model_asset_path=args.model)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        score_threshold=0.5,
        result_callback=visualize_callback,
    )
    detector = vision.ObjectDetector.create_from_options(options)

    # Variables to calculate FPS
    counter, fps = 0, 0
    start_time = time.time()

    # Open webcam video stream
    cap = cv2.VideoCapture(args.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.frame_height)

    while cap.isOpened():

        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        # For frame rate calculation
        counter += 1

        # Flip frame horizontally (like a mirror)
        frame = cv2.flip(frame, 1)

        # Convert frame to RGB format (as expected by MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Run object detection using the model.
        detector.detect_async(mp_image, counter)
        current_frame = mp_image.numpy_view()
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)

        if args.fps:
            # Calculate the FPS
            if counter % FPS_AVG_FRAME_COUNT == 0:
                end_time = time.time()
                fps = FPS_AVG_FRAME_COUNT / (end_time - start_time)
                start_time = time.time()

            show_fps(current_frame, fps)

        # Print result list on pressing 'p'
        if cv2.waitKey(5) & 0xFF == ord("p"):
            print(detection_result_list)

        if detection_result_list:
            # print(detection_result_list)
            annotated_frame = visualize(current_frame, detection_result_list[0])
            cv2.imshow("Object Detection", annotated_frame)
            detection_result_list.clear()
        else:
            cv2.imshow("Object Detection", current_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model",
        help=""""Path of the model. Default: `efficientdet.tflite`. If not found, the
        script will attempt to download it from the default url.
        """,
        required=False,
        default="efficientdet.tflite",
    )

    parser.add_argument(
        "--url",
        help=""""Model url, found on the appropriate page here:
        https://ai.google.dev/edge/mediapipe/solutions
        """,
        required=False,
        default="https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite",
    )

    parser.add_argument(
        "--camera_id", help="Id of camera.", required=False, type=int, default=0
    )

    parser.add_argument(
        "--frame_width",
        help="Width of frame to capture from camera.",
        required=False,
        type=int,
        default=1280,
    )

    parser.add_argument(
        "--frame_height",
        help="Height of frame to capture from camera.",
        required=False,
        type=int,
        default=720,
    )

    parser.add_argument(
        "--fps",
        help="Whether to display the FPS on the screen",
        required=False,
        action="store_true",
    )

    args = parser.parse_args()
    args.camera_id = int(args.camera_id)
    args.model = pathlib.Path(args.model)

    run(args)

