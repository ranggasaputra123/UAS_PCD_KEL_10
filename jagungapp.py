import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# =======================
# KONFIGURASI HALAMAN
# =======================
st.set_page_config(
    page_title="Deteksi Penyakit Jagung",
    layout="wide"
)

# =======================
# LOAD MODEL
# =======================
MODEL_PATH = "JAGUNG/model_jagung.h5"

model = None
if not os.path.exists(MODEL_PATH):
    st.error("Model tidak ditemukan")
else:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# =======================
# KELAS & PENANGGULANGAN
# =======================
CLASSES = ["Healthy", "Common Rust", "Gray Leaf Spot", "Blight"]

TREATMENTS = {
    "Healthy": [
        "Lanjutkan perawatan tanaman secara rutin",
        "Gunakan pupuk berimbang",
        "Pastikan drainase lahan baik"
    ],
    "Common Rust": [
        "Gunakan fungisida mankozeb / propikonazol",
        "Buang daun terinfeksi berat",
        "Lakukan rotasi tanaman"
    ],
    "Gray Leaf Spot": [
        "Gunakan fungisida sistemik",
        "Atur jarak tanam",
        "Gunakan varietas tahan penyakit"
    ],
    "Blight": [
        "Gunakan fungisida klorotalonil",
        "Hindari kelembaban berlebih",
        "Bersihkan sisa tanaman terinfeksi"
    ]
}

# =======================
# FUNGSI PREPROCESS
# =======================
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# =======================
# FUNGSI PREDIKSI
# =======================
def predict_image(img_array):
    preds = model.predict(img_array)
    idx = np.argmax(preds)
    return CLASSES[idx], preds[0][idx]

# =======================
# HALAMAN HOME
# =======================
def home_page():
    st.title("Aplikasi Deteksi Penyakit Daun Jagung")
    st.markdown("""
    Aplikasi ini menggunakan **Deep Learning (CNN)**  
    untuk mendeteksi penyakit daun jagung dari gambar.

    ### Cara Penggunaan
    1. Buka tab **Kamera**
    2. Ambil gambar atau upload foto
    3. Lihat hasil prediksi & penanggulangan
    """)

# =======================
# HALAMAN KAMERA
# =======================
def camera_page():
    st.title("Deteksi Penyakit")

    option = st.radio("Pilih metode input gambar:", ["Kamera", "Upload Gambar"])

    image = None

    if option == "Kamera":
        cam = st.camera_input("Ambil gambar daun jagung")
        if cam:
            image = Image.open(cam)

    else:
        upload = st.file_uploader("Upload gambar daun jagung", type=["jpg", "jpeg", "png"])
        if upload:
            image = Image.open(upload)

    if image:
        st.image(image, caption="Gambar Input", use_container_width=True)
        img_array = preprocess_image(image)
        label, conf = predict_image(img_array)

        st.subheader("Hasil Prediksi")
        st.success(f"Kategori: **{label}**")
        st.write(f"Probabilitas: **{conf:.2%}**")

        st.subheader("Cara Penanggulangan")
        for i, step in enumerate(TREATMENTS[label], 1):
            st.write(f"{i}. {step}")

# =======================
# HALAMAN CONTOH PENYAKIT
# =======================
def example_page():
    st.title("Contoh Penyakit Daun Jagung")

    col1, col2 = st.columns(2)

    with col1:
        st.image("Corn_Health (82).jpg", caption="Healthy")
        st.image("Corn_Common_Rust (2).jpg", caption="Common Rust")

    with col2:
        st.image("Corn_Gray_Spot (3).jpg", caption="Gray Leaf Spot")
        st.image("Corn_Blight (1).jpeg", caption="Blight")

# =======================
# HALAMAN TENTANG
# =======================
def about_page():
    st.title("Tentang Aplikasi")

    st.markdown("""
    Aplikasi ini dikembangkan untuk membantu  
    **deteksi dini penyakit daun jagung**.

    ### Kelompok 10
    **Ketua**
    - M Rangga Saputra

    **Anggota**
    - Iqbal Hidayatullah  
    - M Tegar Yusuf Habibi
    """)

# =======================
# TOPBAR NAVIGATION
# =======================
tabs = st.tabs(["Home", "Kamera", "Contoh Penyakit", "Tentang"])

with tabs[0]:
    home_page()

with tabs[1]:
    camera_page()

with tabs[2]:
    example_page()

with tabs[3]:
    about_page()
