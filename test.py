import os
import cv2
from ultralytics import YOLO
from google.colab.patches import cv2_imshow

def run_prediction(image_name):
    possible_paths = [
        '/content/runs/segment/ayakkabi_final_projesi/weights/best.pt',
        '/content/runs/segment/ayakkabi_kurtarma_operasyonu/weights/best.pt',
        '/content/runs/segment/ayakkabi_final_v7/weights/best.pt'
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path:
        print(f" Model bulundu: {model_path}")
        model = YOLO(model_path)
        
        if not os.path.exists(image_name):
            print(f" HATA: {image_name} bulunamadı! Lütfen fotoğrafı sol panele yükleyin.")
            return

        results = model.predict(source=image_name, conf=0.25, save=True)
        
        res_img_path = os.path.join(results[0].save_dir, image_name)
        img = cv2.imread(res_img_path)
        
        print(f" Test Başarılı! Sonuç kaydedildi: {res_img_path}")
        cv2_imshow(img)
    else:
        print(" HATA: Eğitilmiş model (best.pt) hiçbir klasörde bulunamadı!")

if __name__ == '__main__':
    run_prediction('indir2.jpg')
