import os
import numpy as np
import streamlit as st

from PIL import Image, ImageDraw
import tensorflow as tf
from ultralytics import YOLO

# ===================== PAGE CONFIG & CUSTOM STYLE =====================
st.set_page_config(
    page_title="📊 Stationery Vision Dashboard",
    layout="wide"
)

st.markdown("""
<style>
/* Global background & font */
.stApp {
    background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* GLOBAL TEXT COLOR (abu tua) */
html, body, [data-testid="stAppViewContainer"] {
    color: #2f2f2f !important;
}

.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span, .stMarkdown strong {
    color: #2f2f2f !important;
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    color: #2f2f2f !important;
    font-weight: 700;
}

/* Label pada slider, input, dll */
label, .stSlider label, .stSelectbox label, .stTextInput label {
    color: #2f2f2f !important;
}

/* Sidebar background + font styling */
[data-testid="stSidebar"] {
    background-color: #2f2f4f !important;  /* navy gelap */
    color: #ffffff !important;
}

/* Pastikan teks sidebar SELALU putih */
[data-testid="stSidebar"] * {
    color: #ffffff !important;
    text-shadow: none !important;
}

/* File uploader */
div.stFileUploader > div:first-child {
    background-color: #ffffffdd;
    border-radius: 12px;
    padding: 1rem;
    color: #2f2f2f;
    font-weight: 600;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
}

/* Result card */
.result-box {
    padding: 1rem;
    border-radius: 16px;
    margin-bottom: 1rem;
    text-align: center;
    font-weight: bold;
    color: #2f2f2f;
    box-shadow: 2px 2px 16px rgba(0,0,0,0.2);
}

/* Badge */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    background-color: #2f2f4f;
    color: #ffffff;
    font-size: 0.8rem;
}

/* Tabs rounded */
button[role="tab"] {
    border-radius: 999px !important;
}
button[role="tab"] > div {
    color: #2f2f2f !important;
}

/* ===== SIDEBAR OVERRIDE (PASTIKAN PALING BAWAH) ===== */
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] .stMarkdown span,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: #ffffff !important;
}

</style>
""", unsafe_allow_html=True)

# ===================== SIDEBAR =====================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2921/2921222.png", width=120)
    st.markdown("### 🧷 Stationery Vision Dashboard")
    st.write(
        "Dashboard ini menampilkan dua fitur utama:\n"
        "- Image Classification: Mengklasifikasikan gambar alat tulis (stationery).\n"
        "- Object Detection: Mendeteksi objek alat tulis (stationery) di dalam gambar."
    )
    st.markdown("---")
    st.markdown("Made with ❤️ by Laura Masyitah")

# ===================== LOAD MODELS =====================
@st.cache_resource
def load_classifier_model():
    model_path = os.path.join("models", "model_classification.h5")
    model = tf.keras.models.load_model(model_path)
    return model

@st.cache_resource
def load_detector_model():
    model_path = os.path.join("models", "model_detection.pt")
    model = YOLO(model_path)
    return model

classifier_model = load_classifier_model()
detector_model = load_detector_model()

# Ambil info input dan output model klasifikasi
CLASS_INPUT_SHAPE = classifier_model.input_shape  # (None, H, W, C)
_, CLASS_H, CLASS_W, CLASS_C = CLASS_INPUT_SHAPE
NUM_CLASSES = classifier_model.output_shape[-1]

# Karena label asli tidak diketahui, buat label generik
CLASS_LABELS = [f"Class {i}" for i in range(NUM_CLASSES)]

# Palet warna untuk card/probabilities
PALETTE = [
    "#ffb3ba", "#ffdfba", "#ffffba", "#baffc9",
    "#bae1ff", "#e0bbe4", "#c8b6ff", "#ffd6ff", "#f4b6c2"
]
CLASS_COLORS = {
    CLASS_LABELS[i]: PALETTE[i % len(PALETTE)] for i in range(NUM_CLASSES)
}


# ===================== PREPROCESSING FUNCTIONS =====================
def preprocess_for_classifier(img: Image.Image):
    """
    Preprocess gambar untuk model klasifikasi:
    - Resize ke (CLASS_W, CLASS_H)
    - Convert ke RGB
    - Normalisasi ke [0,1]
    - Tambah dimensi batch -> (1, H, W, C)
    """
    img = img.convert("RGB")
    img = img.resize((CLASS_W, CLASS_H))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# ===================== INFERENCE FUNCTIONS =====================
def predict_class(img: Image.Image):
    """
    Mengembalikan:
    - label (string)
    - confidence (float)
    - probs (numpy array panjang NUM_CLASSES)
    """
    x = preprocess_for_classifier(img)
    preds = classifier_model.predict(x)
    probs = preds[0]
    idx = int(np.argmax(probs))
    label = CLASS_LABELS[idx]
    conf = float(probs[idx])
    return label, conf, probs

def detect_objects_with_yolo(pil_image: Image.Image, model, conf_threshold: float = 0.3):
    """
    Deteksi objek menggunakan YOLO (Ultralytics).
    Mengembalikan:
    - detections: list of dict {box, score, label}
    - image_with_boxes: PIL.Image dengan bounding box
    """
    results = model(pil_image, verbose=False, conf=conf_threshold)[0]

    boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else np.array([])
    conf = results.boxes.conf.cpu().numpy() if results.boxes is not None else np.array([])
    cls = results.boxes.cls.cpu().numpy().astype(int) if results.boxes is not None else np.array([])

    names = model.names  # contoh: {0: '0', 1: '1', 2: '2', 3: '3'}

    detections = []
    for i in range(len(boxes)):
        detections.append({
            "box": boxes[i],                 # [x1, y1, x2, y2]
            "score": float(conf[i]),         # confidence
            "label": names.get(int(cls[i]), f"class_{cls[i]}")
        })

    # Gambar bounding box di atas gambar
    img_draw = pil_image.copy()
    draw = ImageDraw.Draw(img_draw)

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        score = det["score"]

        color = "#32a852"  # hijau stabil

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{label} {score:.2f}"

        text_width = 7 * len(text)
        text_bg = [x1, y1 - 18, x1 + text_width, y1]
        draw.rectangle(text_bg, fill=color)
        draw.text((x1 + 3, y1 - 16), text, fill="white")

    return detections, img_draw

# ===================== MAIN UI =====================
st.title("📊 Stationery Vision Dashboard")
st.write(
    "Upload gambar stationery untuk mencoba **Image Classification** dan "
    "**Object Detection** menggunakan model yang telah dilatih."
)

tab_home, tab_cls, tab_det = st.tabs(["🏠 Home", "📷 Image Classification", "🎯 Object Detection"])

# ---------- HOME TAB ----------
with tab_home:
    st.subheader("Selamat datang! 👋")
    st.write(
        "Gunakan dashboard ini untuk mencoba dua fitur utama:\n"
        "- **Image Classification** untuk mengklasifikasikan gambar alat tulis.\n"
        "- **Object Detection** untuk mendeteksi beberapa objek sekaligus dalam satu gambar."
    )

    sample_path = "sample_images"
    if os.path.isdir(sample_path):
        st.markdown("### 📁 Contoh Gambar")

        # Cari 1 contoh untuk klasifikasi (prefix cls_) dan 1 untuk deteksi (prefix det_)
        exts = (".png", ".jpg", ".jpeg")

        cls_example = None
        det_example = None

        for f in sorted(os.listdir(sample_path)):
            fl = f.lower()
            if fl.startswith("cls_") and fl.endswith(exts) and cls_example is None:
                cls_example = os.path.join(sample_path, f)
            if fl.startswith("det_") and fl.endswith(exts) and det_example is None:
                det_example = os.path.join(sample_path, f)

        col1, col2 = st.columns(2)

        with col1:
            if cls_example is not None:
                img_cls = Image.open(cls_example).convert("RGB")
                img_cls = img_cls.resize((280, 280))  # paksa jadi kotak 280x280
                st.image(img_cls, caption="Sample for Classification", use_container_width=False)
            st.markdown("**📷 Image Classification**")
            st.caption("Upload satu gambar stationery untuk diklasifikasikan dengan model yang telah dilatih.")

        with col2:
            if det_example is not None:
                img_det = Image.open(det_example).convert("RGB")
                img_det = img_det.resize((280, 280))  # paksa jadi kotak 280x280
                st.image(img_det, caption="Sample for Detection", use_container_width=False)
            st.markdown("**🎯 Object Detection**")
            st.caption("Upload gambar yang berisi beberapa stationery untuk dideteksi menggunakan  YOLO.")

    else:
        st.info("Folder `sample_images` belum ditemukan. Pastikan sudah dibuat dan berisi contoh gambar.")


# ---------- IMAGE CLASSIFICATION TAB ----------
with tab_cls:
    st.subheader("📷 Image Classification")
    st.write("Upload gambar stationery untuk diklasifikasikan ke dalam salah satu kelas yang tersedia.")

    uploaded_cls = st.file_uploader(
        "Upload image untuk klasifikasi:",
        type=["jpg", "jpeg", "png"],
        key="cls_uploader"
    )

    if uploaded_cls is not None:
        img = Image.open(uploaded_cls)

        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.image(img, caption="Uploaded Image", use_container_width=True)

        with col2:
            with st.spinner("Melakukan prediksi klasifikasi..."):
                label, conf, probs = predict_class(img)

            color = CLASS_COLORS.get(label, "#ffffff")
            st.markdown(
                f"""
                <div class="result-box" style="background-color:{color};">
                    <h3>Predicted Class</h3>
                    <p style="font-size:28px;">{label} 🎉</p>
                    <span class="badge">Confidence: {conf*100:.2f}%</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("#### 🔎 All Class Probabilities")
            for i, p in enumerate(probs):
                st.write(f"- **{CLASS_LABELS[i]}**: {p*100:.2f}%")

# ---------- OBJECT DETECTION TAB ----------
with tab_det:
    st.subheader("🎯 Object Detection (YOLO)")
    st.write("Upload gambar yang berisi beberapa stationery, model akan mendeteksi objek di dalamnya.")

    uploaded_det = st.file_uploader(
        "Upload image untuk deteksi objek:",
        type=["jpg", "jpeg", "png"],
        key="det_uploader"
    )

    conf_thresh = st.slider(
        "Score threshold (confidence minimum untuk ditampilkan):",
        0.1, 0.9, 0.3, 0.05
    )

    if uploaded_det is not None:
        pil_image = Image.open(uploaded_det).convert("RGB")

        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.image(pil_image, caption="Original Image", use_container_width=True)

        with col2:
            with st.spinner("Melakukan deteksi objek dengan YOLO..."):
                detections, img_with_boxes = detect_objects_with_yolo(
                    pil_image, detector_model, conf_threshold=conf_thresh
                )

            st.image(img_with_boxes, caption="Detection Result", use_container_width=True)

            st.markdown("#### 📋 Detection Results")
            if len(detections) == 0:
                st.info("Tidak ada objek terdeteksi di atas threshold.")
            else:
                for det in detections:
                    x1, y1, x2, y2 = det["box"]
                    st.write(
                        f"- **Label:** {det['label']} "
                        f"| **Confidence:** {det['score']:.2f} "
                        f"| **Box:** ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
                    )
