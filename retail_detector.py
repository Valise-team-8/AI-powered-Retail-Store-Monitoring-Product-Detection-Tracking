import cv2
from ultralytics import YOLO
import math

# Load YOLO model (Medium = more accurate than small)
model = YOLO("yolov8m.pt")  # auto-download if missing

# COCO classes
classNames = model.names

# Retail-specific items commonly found in stores
retail_items = [
    "person", "backpack", "handbag", "suitcase",
    "bottle", "cup", "bowl", "apple", "banana", "orange",
    "broccoli", "carrot", "cake", "donut", "pizza", "sandwich",
    "book", "cell phone", "laptop", "keyboard", "mouse",
    "tvmonitor", "remote", "toaster", "microwave", "oven",
    "refrigerator", "chair", "pottedplant", "teddy bear",
]

# Webcam Setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, stream=True)
    item_count = {}

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])

            # ignore low confidence
            if conf < 0.55:
                continue

            cls = int(box.cls[0])
            class_name = classNames[cls]

            if class_name in retail_items:
                item_count[class_name] = item_count.get(class_name, 0) + 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                cv2.putText(
                    frame, f"{class_name} {conf:.2f}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2
                )

    # 🧾 Display Count Panel
    panel_y = 20
    cv2.putText(frame, "Retail Items Detected:", (10, panel_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    for item, count in item_count.items():
        panel_y += 25
        cv2.putText(frame, f"{item}: {count}", (10, panel_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("Retail Monitoring - YOLOv8", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

