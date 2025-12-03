from ultralytics import YOLO
from PIL import Image
import os

model_path = "models/model_detection.pt"
img_path = os.path.join("sample_images", "det_1.jpg")  # ganti ke .jpg kalau perlu

# 1. Load YOLO model
model = YOLO(model_path)
print("✅ YOLO model loaded successfully!")

# 2. Cek gambar
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Gambar contoh tidak ditemukan: {img_path}")

# 3. Inference
results = model(img_path, verbose=False)  # 1 gambar
result = results[0]

print("\n===== HASIL DETEKSI =====")
boxes = result.boxes

# Boxes dalam format xyxy (x1,y1,x2,y2)
xyxy = boxes.xyxy.cpu().numpy()
conf = boxes.conf.cpu().numpy()
cls = boxes.cls.cpu().numpy().astype(int)

print(f"Jumlah objek terdeteksi: {len(xyxy)}")

# Nama kelas dari model
names = model.names  # dict: {0: '0', 1: '1', 2: '2', 3: '3'}

for i in range(len(xyxy)):
    x1, y1, x2, y2 = xyxy[i]
    score = conf[i]
    class_id = cls[i]
    class_name = names.get(class_id, f"class_{class_id}")
    print(
        f"- {class_name} (id={class_id}), "
        f"conf={score:.2f}, "
        f"box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
    )
