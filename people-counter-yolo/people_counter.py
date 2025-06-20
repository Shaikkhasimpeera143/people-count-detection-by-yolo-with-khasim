from ultralytics import YOLO
import cv2

# Load YOLOv8 model (nano version - fast)
model = YOLO("yolov8n.pt")

# Load your image (change filename if needed)
img_path = "your_image.jpg"
img = cv2.imread(img_path)

# Run YOLO detection
results = model(img)

# Initialize people count
people_count = 0

# Loop through detected objects
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    if cls_id == 0 and conf > 0.5:  # class 0 = person
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, "Person", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        people_count += 1

# Show total people count on image
cv2.putText(img, f"People Count: {people_count}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Show result
cv2.imshow("People Counter", img)
cv2.imwrite("output_people_count.jpg", img)
cv2.waitKey(0)

cv2.destroyAllWindows()