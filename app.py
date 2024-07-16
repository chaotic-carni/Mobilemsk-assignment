from flask import Flask, Response, render_template, stream_with_context
import cv2
from ultralytics import YOLO
import numpy as np
import torch
from keras._tf_keras.keras.models import load_model
import time

model1 = load_model(r"model.h5")
app = Flask(__name__)

if torch.cuda.is_available():
    print("Yayy GPU is available and potentially being used by Ultralytics.")
else:
    print("GPU is not available or not being used by Ultralytics.")

model_pred = YOLO("yolov8n-pose.pt")

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Initialize global variable
predicted_class = None


def extract_pose_lines(original, pose_output):
    # Function to extract pose lines from the pose detection output
    assert original.shape == pose_output.shape, "Images must be the same size"
    diff = cv2.absdiff(pose_output, original)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
    transparent = np.zeros((original.shape[0], original.shape[1], 4), dtype=np.uint8)
    transparent[:,:,0] = cv2.bitwise_and(pose_output[:,:,0], mask)
    transparent[:,:,1] = cv2.bitwise_and(pose_output[:,:,1], mask)
    transparent[:,:,2] = cv2.bitwise_and(pose_output[:,:,2], mask)
    transparent[:,:,3] = mask
    return transparent

def extract_keypoints(results):
    keypoints_list = []
    for result in results:
        keypoints = result.keypoints.xyn
        keypoints_list.append(keypoints.cpu().numpy())
    return keypoints_list

def get_class_name(num):
    if len(num) == 1:
        if num == 1:
            return "Adho Mukha"
        elif num == 2:
            return "Goddess"
        elif num == 3:
            return "Plank"
        elif num == 4:
            return "Standing"
        elif num == 5:
            return "Warrior"
    return ""

def generate_frames():
    frame_count = 0
    last_pose_lines = None
    global predicted_class
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame_count = (frame_count + 1) % 3
            
            if frame_count == 0:
                results = model_pred(frame)
                keypoints = extract_keypoints(results)
                try:
                    prediction = model1.predict(keypoints)
                    predicted_class = np.argmax(prediction, axis=1)
                except:
                    print("model error")
                annotated_frame = results[0].plot()
                last_pose_lines = extract_pose_lines(frame, annotated_frame)
            else:
                annotated_frame = frame.copy()
            if last_pose_lines is not None:
                if last_pose_lines.shape[:2] != annotated_frame.shape[:2]:
                    last_pose_lines = cv2.resize(last_pose_lines, (annotated_frame.shape[1], annotated_frame.shape[0]))
                mask = last_pose_lines[:,:,3] / 255.0
                for c in range(0, 3):
                    annotated_frame[:,:,c] = annotated_frame[:,:,c] * (1 - mask) + last_pose_lines[:,:,c] * mask
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/class_feed')
def class_feed():
    def generate():
        while True:
            if predicted_class is not None:
                class_name = get_class_name(predicted_class)
                yield f"data: {class_name}\n\n"
            time.sleep(0.1)
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)
