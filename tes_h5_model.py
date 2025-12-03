import tensorflow as tf
import numpy as np
from PIL import Image
import os

# 1. Load model
model = tf.keras.models.load_model("models/model_classification.h5")
print("Model loaded successfully!")

# 2. Cek input shape
input_shape = model.input_shape  # (None, 128, 128, 3)
print("Model input shape:", input_shape)

_, H, W, C = input_shape  # -> H=128, W=128, C=3

# 3. Path gambar contoh
img_path = os.path.join("sample_images", "cls_1.jpeg")

if not os.path.exists(img_path):
    raise FileNotFoundError(f"Gambar contoh tidak ditemukan: {img_path}")

# 4. Load & preprocess gambar sesuai input model
img = Image.open(img_path)

if C == 1:
    img = img.convert("L")   # grayscale
else:
    img = img.convert("RGB") # 3 channel

img = img.resize((W, H))     # (128, 128)

arr = np.array(img) / 255.0  # normalisasi 0-1

if C == 1:
    arr = np.expand_dims(arr, axis=-1)

arr = np.expand_dims(arr, axis=0)   # (1, 128, 128, 3)

print("Input array shape:", arr.shape)

# 5. Predict
pred = model.predict(arr)
print("Raw prediction:", pred)
print("Predicted class index:", int(np.argmax(pred)))
