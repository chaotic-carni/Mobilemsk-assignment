from flask import Flask, Response, render_template
import cv2
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO('yolov8n-pose.pt')

# Initialize the video capture
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Run YOLOv8 inference on the frame
            results = model(frame)
            
            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            
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