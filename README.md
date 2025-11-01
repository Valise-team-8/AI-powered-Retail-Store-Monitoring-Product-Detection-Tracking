# AI-Powered Retail Store Monitoring

This project implements real-time object detection for retail store monitoring using YOLOv8 computer vision models. It can detect various retail items and people in a live video feed from a webcam.

## Features

- **Real-time Detection**: Uses YOLOv8 to detect objects in live webcam feed
- **Retail-Focused**: Specifically trained to recognize common retail items like products, furniture, and people
- **Web Interface**: Flask-based web application with live video streaming
- **Desktop Mode**: Standalone OpenCV window for local monitoring
- **Statistics Panel**: Displays counts of detected objects in real-time

## Detected Objects

The system can detect the following retail-related items:
- People and accessories (backpack, handbag, suitcase)
- Food items (apple, banana, orange, broccoli, carrot, cake, donut, pizza, sandwich)
- Beverages (bottle, cup, bowl)
- Electronics (cell phone, laptop, keyboard, mouse, TV monitor, remote)
- Appliances (toaster, microwave, oven, refrigerator)
- Furniture (chair)
- Other (book, potted plant, teddy bear)

## Requirements

- Python 3.8+
- Webcam
- Required packages:
  - Flask
  - OpenCV (cv2)
  - Ultralytics (YOLOv8)

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install flask opencv-python ultralytics
   ```

## Usage

### Web Application (Recommended)

Run the Flask web server:

```bash
python app.py
```

Open your browser and navigate to `http://localhost:5000` to view the monitoring dashboard.

### Desktop Application

For local monitoring without a web server:

```bash
python retail_detector.py
```

A window will open showing the live feed with detected objects and counts.

## Configuration

- **Model Selection**: Change `MODEL_PATH` in `app.py` to use different YOLO models:
  - `yolov8n.pt` (nano - fastest, least accurate)
  - `yolov8s.pt` (small - balanced)
  - `yolov8m.pt` (medium - accurate, default)
- **Confidence Threshold**: Adjust `CONFIDENCE_THRESHOLD` in `app.py` (default: 0.55)

## How It Works

1. Captures video frames from the default webcam
2. Runs YOLOv8 object detection on each frame
3. Filters detections to only include retail-relevant items
4. Draws bounding boxes and labels on detected objects
5. Maintains running counts of detected items
6. Streams processed video and statistics to the web interface

## Troubleshooting

- **Webcam not found**: Ensure your webcam is connected and not in use by other applications
- **Slow performance**: Try using a smaller YOLO model (yolov8n.pt or yolov8s.pt)
- **Low detection accuracy**: Increase confidence threshold or use a larger model (yolov8m.pt)

## License

This project uses YOLOv8 models from Ultralytics. Please refer to their licensing terms.