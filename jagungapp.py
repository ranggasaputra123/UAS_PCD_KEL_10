import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# =======================
# KONFIGURASI HALAMAN
# =======================
st.set_page_config(
    page_title="Deteksi",
    layout="wide"
)

# =======================
# LOAD MODEL
# =======================
MODEL_PATH = "JAGUNG/model_jagung.h5"

model = None
if not os.path.exists(MODEL_PATH):
    st.error(f"Model tidak ditemukan di {MODEL_PATH}")
else:
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        st.success("Model berhasil dimuat")
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")

# =======================
# KELAS PENYAKIT
# =======================
CLASSES = ["Healthy", "Common Rust", "Gray Leaf Spot", "Blight"]

# =======================
# DATA PENANGGULANGAN
# =======================
TREATMENTS = {
    "Healthy": [
        "Lanjutkan perawatan tanaman secara rutin.",
        "Gunakan pupuk berimbang.",
        "Pastikan sistem drainase lahan baik."
    ],
    "Common Rust": [
        "Gunakan fungisida berbahan aktif mankozeb atau propikonazol.",
        "Buang dan musnahkan daun yang terinfeksi berat.",
        "Lakukan rotasi tanaman."
    ],
    "Gray Leaf Spot": [
        "Semprot fungisida sistemik sesuai dosis anjuran.",
        "Atur jarak tanam agar sirkulasi udara baik.",
        "Gunakan varietas jagung tahan penyakit."
    ],
    "Blight": [
        "Gunakan fungisida berbahan aktif klorotalonil.",
        "Hindari penyiraman berlebih.",
        "Bersihkan sisa tanaman yang terinfeksi dari lahan."
    ]
}

# =======================
# FUNGSI PREPROCESS
# =======================
def preprocess_image(img):
    try:
        img = img.convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing gambar: {e}")
        return None

# =======================
# FUNGSI PREDIKSI
# =======================
def predict_image(img_array):
    try:
        preds = model.predict(img_array)
        class_idx = np.argmax(preds, axis=1)[0]
        confidence = preds[0][class_idx]
        return CLASSES[class_idx], confidence
    except Exception as e:
        st.error(f"Error prediksi: {e}")
        return None, None

# =======================
# HALAMAN HOME
# =======================
def home_page():
    st.title("Aplikasi Deteksi Penyakit Daun Jagung")
    st.markdown("""
    Aplikasi ini digunakan untuk mendeteksi **penyakit pada daun jagung**
    menggunakan teknologi **kecerdasan buatan (AI)** berbasis *Deep Learning*.

    ### Cara Penggunaan
    1. Buka tab **Kamera**
    2. Ambil gambar daun jagung
    3. Lihat hasil prediksi dan cara penanggulangan

    ### Kategori Penyakit
    - **Healthy** (Sehat)
    - **Common Rust**
    - **Gray Leaf Spot**
    - **Blight**
    """)

# =======================
# HALAMAN KAMERA
# =======================
def camera_page():
    st.title("Deteksi Penyakit Melalui Kamera")

    if model is None:
        st.error("Model belum dimuat.")
        return

    camera_input = st.camera_input("Ambil gambar daun jagung")

    if camera_input is not None:
        st.image(camera_input, caption="Gambar Daun Jagung", use_container_width=True)

        img = Image.open(camera_input)
        img_array = preprocess_image(img)

        if img_array is not None:
            label, confidence = predict_image(img_array)

            if label:
                st.subheader("Hasil Prediksi")
                st.success(f"Kategori: **{label}**")
                st.write(f"Probabilitas: **{confidence:.2%}**")

                st.subheader("Cara Penanggulangan")
                with st.expander("Klik untuk melihat penanggulangan"):
                    for i, step in enumerate(TREATMENTS[label], 1):
                        st.write(f"{i}. {step}")

# =======================
# HALAMAN TENTANG
# =======================
def about_page():
    st.title("Tentang Aplikasi")

    st.markdown("""
    Aplikasi **Deteksi Penyakit Daun Jagung** memanfaatkan teknologi
    *Convolutional Neural Network (CNN)* untuk mengklasifikasikan
    kondisi daun jagung secara otomatis melalui citra kamera.
    """)

    st.header("Kelompok 10")
    st.markdown("""
    **Ketua Kelompok**
    - M Rangga Saputra

    **Anggota**
    - Iqbal Hidayatullah  
    - M Tegar Yusuf Habibi
    """)

    st.subheader("Tujuan")
    st.markdown("""
    Mengembangkan sistem deteksi penyakit daun jagung yang **akurat,
    cepat, dan efisien** untuk membantu petani meminimalisir risiko
    gagal panen.
    """)

# =======================
# TOPBAR NAVIGATION
# =======================
tabs = st.tabs(["Home", "Kamera", "Tentang Aplikasi"])

with tabs[0]:
    home_page()

with tabs[1]:
    camera_page()

with tabs[2]()
