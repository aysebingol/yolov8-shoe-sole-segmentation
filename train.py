
from ultralytics import YOLO
from roboflow import Roboflow


rf = Roboflow(api_key="PO3Z3JYlkHRGl6kXy8UY")
project = rf.workspace("yolov5-q03r3").project("yolo8-shoe-sole")
version = project.version(1)
dataset = version.download("yolov8")

model = YOLO('yolov8n-seg.pt')
model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=50,
    imgsz=640,
    plots=True,
    name='ayakkabi_taban_projesi'
)
