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
    df = df.rename(columns={
        'Suhu Udara': 'Tavg: Temperatur rata-rata (°C)',
        'Kelembapan Udara': 'RH_avg: Kelembapan rata-rata (%)',
        'Curah Hujan/Jam': 'RR: Curah hujan (mm)',
        'Kecepatan Angin (ms)': 'ff_avg: Kecepatan angin rata-rata (m/s)',
        'Kelembapan Tanah': 'Kelembaban Permukaan Tanah',
        'Waktu': 'Waktu'
    })

    fitur = [
        'Tavg: Temperatur rata-rata (°C)',
        'RH_avg: Kelembapan rata-rata (%)',
        'RR: Curah hujan (mm)',
        'ff_avg: Kecepatan angin rata-rata (m/s)',
        'Kelembaban Permukaan Tanah'
    ]

    raw_last_row = df.iloc[-1]
    clean_dict = {}
    for col in fitur:
        try:
            val = str(raw_last_row[col]).replace(',', '.')
            val = float(val)
        except:
            val = 0.0
        clean_dict[col] = val

    clean_input_df = pd.DataFrame([clean_dict])
    scaled = scaler.transform(clean_input_df)
    prediction_label = convert_to_label(model.predict(scaled)[0])

    df.at[raw_last_row.name, 'Prediksi Kebakaran'] = prediction_label
    last_row = df.loc[raw_last_row.name]

    waktu = pd.to_datetime(last_row['Waktu'])
    hari = convert_day_to_indonesian(waktu.strftime('%A'))
    bulan = convert_month_to_indonesian(waktu.strftime('%B'))
    tanggal = waktu.strftime(f'%d {bulan} %Y')
    font, bg = risk_styles.get(prediction_label, ("black", "white"))

    sensor_df = pd.DataFrame({
        "Variabel": fitur,
        "Value": [f"{clean_dict[col]:.1f}" for col in fitur]
    })
    st.write("Data Sensor Realtime:")
    st.table(sensor_df)

    st.markdown(
        f"<p style='background-color:{bg}; color:{font}; padding:10px; border-radius:8px; font-weight:bold;'>"
        f"Pada hari {hari}, tanggal {tanggal}, lahan ini diprediksi memiliki tingkat resiko kebakaran: "
        f"<span style='text-decoration: underline; font-size: 22px;'>{prediction_label}</span></p>",
        unsafe_allow_html=True
    )
