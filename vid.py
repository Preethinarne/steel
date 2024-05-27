import cv2
import pandas as pd
from ultralytics import YOLO

# Load the pre-trained YOLO model
model = YOLO('best.pt')

# Read the COCO class list from a file
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Open a video capture object
video_path = 0  # Change this to the path of your video file

cap = cv2.VideoCapture(video_path)

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Resize the frame to a fixed size (if necessary)
    frame = cv2.resize(frame, (1020, 500))

    # Perform object detection on the frame
    results = model.predict(frame)
    detections = results[0].boxes.data
    px = pd.DataFrame(detections).astype("float")

    # Iterate through detections and draw bounding boxes
    for index, row in px.iterrows():
        x1, y1, x2, y2, _, d = map(int, row)
        c = class_list[d]

        # Draw bounding boxes and class labels on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

    # Display the frame with objects detected
    cv2.imshow("Object Detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()