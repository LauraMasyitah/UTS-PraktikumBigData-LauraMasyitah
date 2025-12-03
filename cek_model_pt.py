from ultralytics import YOLO

model_path = "models/model_detection.pt"

# 1. Load model YOLO
model = YOLO(model_path)
print("✅ YOLO model loaded successfully!")
print("Tipe objek model:", type(model))

print("\n===== INFO MODEL =====")
model.info()  # ringkasan arsitektur

# Opsional: cek nama class kalau ada
if model.names:
    print("\n===== CLASS NAMES (model.names) =====")
    print(model.names)
