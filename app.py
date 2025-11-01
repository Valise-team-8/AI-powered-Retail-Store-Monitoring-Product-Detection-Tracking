from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import math
from threading import Lock
import atexit

app = Flask(__name__)

# --- Configuration ---
MODEL_PATH = "yolov8m.pt"
CONFIDENCE_THRESHOLD = 0.55
# ---

model = YOLO(MODEL_PATH)
classNames = model.names

# Use a set for faster O(1) lookups
retail_items = {
    "person","backpack","handbag","suitcase","bottle","cup",
    "bowl","apple","banana","orange","broccoli","carrot",
    "cake","donut","pizza","sandwich","book","cell phone",
    "laptop","keyboard","mouse","tvmonitor","remote",
    "toaster","microwave","oven","refrigerator","chair",
    "pottedplant","teddy bear"
}

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam. Please check if it is connected and not in use by another application.")


# Thread-safe data sharing
detected_counts = {}
data_lock = Lock()

def gen_frames():
    global detected_counts
    while True:
        success, frame = cap.read()
        if not success:
            # If reading fails, wait a bit before retrying
            cv2.waitKey(100)
            break
            # If reading from the camera fails, end the stream.
            break 

        results = model(frame, stream=True)
        current_frame_counts = {}

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < CONFIDENCE_THRESHOLD: continue

                cls = int(box.cls[0])
                name = classNames[cls]

                if name in retail_items:
                    current_frame_counts[name] = current_frame_counts.get(name, 0) + 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(frame,f"{name} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
        
        with data_lock:
            detected_counts = current_frame_counts

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    with data_lock:
        # Return a copy to avoid issues if it's modified while being sent
        return jsonify(detected_counts.copy())

if __name__ == "__main__":
    # Register a cleanup function to be called on application exit
    atexit.register(lambda: cap.release())
    app.run(debug=True)
