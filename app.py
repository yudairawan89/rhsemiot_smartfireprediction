import streamlit as st
st.set_page_config(page_title="Prediksi Kebakaran RHSEM-IoT", layout="centered")

import pandas as pd
import joblib
import numpy as np

# ==== Load model dan scaler ====
@st.cache_resource
def load_model():
    return joblib.load("RHSEM_IoT_Model.joblib")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.joblib")

model = load_model()
scaler = load_scaler()

# ==== Mapping label prediksi ====
label_mapping = {
    0: "Low / Rendah",
    1: "Moderate / Sedang",
    2: "High / Tinggi",
    3: "Very High / Sangat Tinggi"
}

color_mapping = {
    "Low / Rendah": ("white", "blue"),
    "Moderate / Sedang": ("white", "green"),
    "High / Tinggi": ("black", "yellow"),
    "Very High / Sangat Tinggi": ("white", "red")
}

# ==== Judul Aplikasi ====
st.title("🔥 Prediksi Risiko Kebakaran - RHSEM IoT 🔥")
st.markdown("Masukkan nilai sensor berikut untuk memprediksi tingkat risiko kebakaran.")

# ==== Input User ====
col1, col2 = st.columns(2)
with col1:
    suhu = st.number_input("Suhu Udara (°C)", value=30.0)
    kelembapan = st.number_input("Kelembapan Udara (%)", value=65.0)
    curah_hujan = st.number_input("Curah Hujan (mm)", value=5.0)
with col2:
    angin = st.number_input("Kecepatan Angin (m/s)", value=2.0)
    kelembapan_tanah = st.number_input("Kelembaban Tanah (%)", value=50.0)

# ==== Proses Prediksi ====
if st.button("Prediksi Risiko Kebakaran"):
    input_df = pd.DataFrame({
        'Tavg: Temperatur rata-rata (°C)': [suhu],
        'RH_avg: Kelembapan rata-rata (%)': [kelembapan],
        'RR: Curah hujan (mm)': [curah_hujan],
        'ff_avg: Kecepatan angin rata-rata (m/s)': [angin],
        'Kelembaban Permukaan Tanah': [kelembapan_tanah]
    })

    scaled_input = scaler.transform(input_df)
    pred = model.predict(scaled_input)[0]
    label = label_mapping.get(pred, "Unknown")

    font_color, bg_color = color_mapping.get(label, ("black", "white"))

    st.markdown(
        f"<div style='background-color:{bg_color}; color:{font_color}; padding:20px; border-radius:10px;'>"
        f"<h3 style='text-align:center;'>Prediksi Risiko Kebakaran:</h3>"
        f"<h1 style='text-align:center; text-decoration:underline;'>{label}</h1>"
        f"</div>",
        unsafe_allow_html=True
    )
