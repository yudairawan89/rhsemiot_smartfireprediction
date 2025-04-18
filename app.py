import streamlit as st
st.set_page_config(page_title="Smart Fire Prediction RHSEM – IoT", layout="wide")

import pandas as pd
import joblib
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# === AUTO REFRESH ===
st_autorefresh(interval=3000, key="data_refresh")

# === STYLE KUSTOM ===
st.markdown("""
    <style>
    .main {background-color: #F9F9F9;}
    table {width: 100%; border-collapse: collapse;}
    th, td {border: 1px solid #ddd; padding: 8px;}
    th {background-color: #e0e0e0; text-align: center;}
    td {text-align: center;}
    .section-title {
        background-color: #1f77b4;
        color: white;
        padding: 10px;
        border-radius: 6px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# === FUNGSI PENDUKUNG ===
def convert_day_to_indonesian(day_name):
    return {'Monday': 'Senin', 'Tuesday': 'Selasa', 'Wednesday': 'Rabu',
            'Thursday': 'Kamis', 'Friday': 'Jumat', 'Saturday': 'Sabtu',
            'Sunday': 'Minggu'}.get(day_name, day_name)

def convert_month_to_indonesian(month_name):
    return {'January': 'Januari', 'February': 'Februari', 'March': 'Maret',
            'April': 'April', 'May': 'Mei', 'June': 'Juni', 'July': 'Juli',
            'August': 'Agustus', 'September': 'September', 'October': 'Oktober',
            'November': 'November', 'December': 'Desember'}.get(month_name, month_name)

def convert_to_label(pred):
    return {0: "Low", 1: "Moderate", 2: "High", 3: "Very High"}.get(pred, "Unknown")

risk_styles = {
    "Low": ("white", "blue"),
    "Moderate": ("white", "green"),
    "High": ("black", "yellow"),
    "Very High": ("white", "red")
}

# === LOAD MODEL DAN SCALER ===
@st.cache_resource
def load_model():
    return joblib.load("RHSEM_IoT_Model.joblib")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.joblib")

model = load_model()
scaler = load_scaler()

# === BACA DATA TANPA CACHE ===
def load_data():
    url = "https://docs.google.com/spreadsheets/d/1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM/export?format=csv"
    return pd.read_csv(url)

# force load terbaru
st.cache_data.clear()
df = load_data()

# === HEADER ===
col1, col2 = st.columns([1, 9])
with col1:
    st.image("logo.png", width=90)
with col2:
    st.markdown("<h2 style='margin-bottom:0;'>Smart Fire Prediction RHSEM – IoT Model</h2>", unsafe_allow_html=True)
    col_deskripsi, col_btn = st.columns([8, 2])
    with col_deskripsi:
        st.caption("Sistem Prediksi Tingkat Resiko Kebakaran Hutan dan Lahan menggunakan model Hybrid Machine & Deep Learning. Data diambil dari IoT secara Realtime via Google Sheets.")
    with col_btn:
        st.markdown(
            "<a href='https://docs.google.com/spreadsheets/d/1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM/edit?gid=0#gid=0' target='_blank'>"
            "<button style='padding: 6px 16px; background-color: #1f77b4; color: white; border: none; border-radius: 4px; cursor: pointer;'>Data Cloud</button>"
            "</a>",
            unsafe_allow_html=True
        )

# === PREDIKSI REALTIME ===
st.markdown("<div class='section-title'>Hasil Prediksi Data Realtime</div>", unsafe_allow_html=True)

if df is not None and not df.empty:
    ...  # Existing realtime prediction logic

    # === PREDIKSI MANUAL ===
    st.markdown("<div class='section-title'>Pengujian Menggunakan Data Manual</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        suhu = st.number_input("Suhu Udara (°C)", value=30.0)
        kelembapan = st.number_input("Kelembapan Udara (%)", value=65.0)
    with col2:
        curah = st.number_input("Curah Hujan (mm)", value=10.0)
        angin = st.number_input("Kecepatan Angin (m/s)", value=3.0)
    with col3:
        tanah = st.number_input("Kelembaban Tanah (%)", value=50.0)

    if st.button("Prediksi Manual"):
        input_df = pd.DataFrame([{
            'Tavg: Temperatur rata-rata (°C)': suhu,
            'RH_avg: Kelembapan rata-rata (%)': kelembapan,
            'RR: Curah hujan (mm)': curah,
            'ff_avg: Kecepatan angin rata-rata (m/s)': angin,
            'Kelembaban Permukaan Tanah': tanah
        }])
        scaled_manual = scaler.transform(input_df)
        manual_pred = convert_to_label(model.predict(scaled_manual)[0])
        fnt, bgc = risk_styles.get(manual_pred, ("black", "white"))
        st.markdown(
            f"<p style='color:{fnt}; background-color:{bgc}; padding:10px; border-radius:5px;'>"
            f"Prediksi Risiko Kebakaran: <b>{manual_pred}</b></p>",
            unsafe_allow_html=True
        )

# === FOOTER ===
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<center>
<h4>Smart Fire Prediction RHSEM – IoT Model</h4>
<p>Dikembangkan oleh Mahasiswa Universitas Putera Indonesia YPTK Padang Tahun 2025</p>
</center>
""", unsafe_allow_html=True)
