from flask import Flask, Response, render_template
import cv2
from ultralytics import YOLO
import numpy as np
import torch
from keras._tf_keras.keras.models import load_model

model1 = load_model(r"model.h5")

app = Flask(__name__)

if torch.cuda.is_available():
  print("Yayy GPU is available and potentially being used by Ultralytics.")
else:
  print("GPU is not available or not being used by Ultralytics.")
model_pred = YOLO("yolov8n-pose.pt")

# Initialize the video capture
cap = cv2.VideoCapture(0)

def extract_pose_lines(original, pose_output):
    """
    Extracts pose lines from the pose detection output and returns them on a transparent background.

    Args:
    original (np.array): The original image.
    pose_output (np.array): The pose detection output image.

    Returns:
    np.array: An RGBA image with pose lines on a transparent background.
    """
    # Ensure both images are the same size
    assert original.shape == pose_output.shape, "Images must be the same size"

    # Calculate the absolute difference for each color channel
    diff = cv2.absdiff(pose_output, original)

    # Convert the difference to grayscale
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary mask
    _, mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)

    # Create a transparent background
    transparent = np.zeros((original.shape[0], original.shape[1], 4), dtype=np.uint8)

    # Copy the colored pose lines to the transparent image
    transparent[:,:,0] = cv2.bitwise_and(pose_output[:,:,0], mask)
    transparent[:,:,1] = cv2.bitwise_and(pose_output[:,:,1], mask)
    transparent[:,:,2] = cv2.bitwise_and(pose_output[:,:,2], mask)
    transparent[:,:,3] = mask  # Use the mask for the alpha channel

    return transparent

def extract_keypoints(results):
    keypoints_list = []
    for result in results:
        keypoints = result.keypoints.xyn
        keypoints_list.append(keypoints.cpu().numpy())
    return keypoints_list

def print_class(num):
    if len(num) == 1:
        if num == 1:
            print("downdog")
        elif num ==2 :
            print("goddess")
        elif num ==3 :
            print("plank")
        elif num ==4 :
            print("standing")
        elif num ==5 :
            print("warrior")

def generate_frames():
    frame_count = 0
    last_pose_lines = None
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame_count = (frame_count + 1) % 3
            
            if frame_count == 0:
                # Run YOLOv8 inference on every 3rd frame
                results = model_pred(frame)
                keypoints = extract_keypoints(results)
                prediction = model1.predict(keypoints)
                predicted_class = np.argmax(prediction, axis=1)
                print_class(predicted_class)
                
                # Visualize the results on the frame
                annotated_frame = results[0].plot()
                
                # Extract pose lines
                last_pose_lines = extract_pose_lines(frame, annotated_frame)
            else:
                # Use the original frame for other frames
                annotated_frame = frame.copy()
            
            # Overlay the last extracted pose lines if available
            if last_pose_lines is not None:
                # Ensure last_pose_lines and annotated_frame have the same size
                if last_pose_lines.shape[:2] != annotated_frame.shape[:2]:
                    last_pose_lines = cv2.resize(last_pose_lines, (annotated_frame.shape[1], annotated_frame.shape[0]))
                
                # Create a mask from the alpha channel of last_pose_lines
                mask = last_pose_lines[:,:,3] / 255.0
                
                # Blend the pose lines with the frame
                for c in range(0, 3):
                    annotated_frame[:,:,c] = annotated_frame[:,:,c] * (1 - mask) + last_pose_lines[:,:,c] * mask
            
            # Encode the frame to JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
