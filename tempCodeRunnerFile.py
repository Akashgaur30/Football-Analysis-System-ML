#! C:\Users\Dell\Desktop\footballAI\venv\Scripts\python.exe
from ultralytics import YOLO


model = YOLO('yolov8s')

results = model.predict('input_videos/foo.mp4',save=True)
print(results[0])
print('=========================')
for box in results[0].boxes:
    print(box)