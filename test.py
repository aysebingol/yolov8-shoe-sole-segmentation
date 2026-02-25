
from ultralytics import YOLO
import os

model = YOLO('best.pt') 


results = model.predict(source='test/images', save=True, conf=0.5)

print("Test işlemi tamamlandı, sonuçlar runs/segment/predict klasöründe.")
