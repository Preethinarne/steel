import cv2
import pandas as pd
from ultralytics import YOLO

# Load the pre-trained YOLO model
model = YOLO('best.pt')

# Load an image
frame = cv2.imread('1.jpg')

# Resize the frame to a fixed size (if necessary)
frame = cv2.resize(frame, (1020, 500))

# Perform object detection on the frame
results = model.predict(frame)

detections = results[0].boxes.data  
px = pd.DataFrame(detections).astype("float")
print(px)

# Read the COCO class list from a file
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

for index, row in px.iterrows():
    x1 = int(row[0])
    y1 = int(row[1])
    x2 = int(row[2])
    y2 = int(row[3])
    d = int(row[5])
    c = class_list[d]
    print(c)

    # Draw bounding boxes and class labels on the frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

# Display the frame with objects detected
cv2.imshow("Object Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
