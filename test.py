from ultralytics import YOLO

model = YOLO('best.pt')

model.predict('1.jpg', conf=0.1, show=True, save=True)