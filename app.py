import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_autorefresh import st_autorefresh
from streamlit_folium import folium_static
from datetime import datetime

# Konfigurasi halaman
st.set_page_config(page_title="Smart Fire Prediction RHSEM - IoT", layout="wide")

# Auto-refresh setiap 3 detik
st_autorefresh(interval=3000, key="data_refresh")

# Style CSS sederhana
st.markdown("""
    <style>
    .section-title {
        background-color: #1f77b4;
        color: white;
        padding: 10px;
        border-radius: 6px;
        font-weight: bold;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# === Fungsi bantu ===
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

# === Load model & scaler ===
@st.cache_resource
def load_model():
    return joblib.load("RHSEM_IoT_Model.joblib")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.joblib")

model = load_model()
scaler = load_scaler()

# === Load data dari Google Sheets ===
@st.cache_data(ttl=10)
def load_data():
    url = "https://docs.google.com/spreadsheets/d/1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM/export?format=csv"
    return pd.read_csv(url)

df = load_data()

# === Judul Header ===
st.title("Smart Fire Prediction RHSEM - IoT")
st.markdown("Prediksi kebakaran hutan dan lahan secara real-time berbasis IoT dan model ensemble learning.")

# === Section: Hasil Prediksi Realtime ===
st.markdown("<div class='section-title'>Hasil Prediksi Data Realtime</div>", unsafe_allow_html=True)

if df is not None and not df.empty:
    # Rename kolom agar sesuai dengan fitur
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

    # Preprocessing
    clean_df = df[fitur].copy()
    for col in fitur:
        clean_df[col] = clean_df[col].astype(str).str.replace(',', '.').astype(float).fillna(0)

    # Prediksi
    scaled_all = scaler.transform(clean_df)
    df["Prediksi Kebakaran"] = [convert_to_label(p) for p in model.predict(scaled_all)]

    # Data terbaru
    last_row = df.iloc[-1]
    waktu = pd.to_datetime(last_row['Waktu'])
    hari = convert_day_to_indonesian(waktu.strftime('%A'))
    bulan = convert_month_to_indonesian(waktu.strftime('%B'))
    tanggal = waktu.strftime(f'%d {bulan} %Y')
    label = last_row["Prediksi Kebakaran"]
    font, bg = risk_styles.get(label, ("black", "white"))

    # Tabel data sensor
    sensor_df = pd.DataFrame({
        "Parameter": ["Suhu", "Kelembapan", "Curah Hujan", "Angin", "Kelembaban Tanah"],
        "Nilai": [
            f"{last_row[fitur[0]]} °C",
            f"{last_row[fitur[1]]} %",
            f"{last_row[fitur[2]]} mm",
            f"{last_row[fitur[3]]} m/s",
            f"{last_row[fitur[4]]} %"
        ]
    })
    st.write("Data Sensor (Realtime):")
    st.table(sensor_df)

    # Hasil Prediksi
    st.markdown(
        f"<p style='background-color:{bg}; color:{font}; padding:10px; border-radius:8px; font-weight:bold;'>"
        f"Pada hari {hari}, {tanggal}, prediksi tingkat risiko kebakaran adalah: "
        f"<span style='font-size: 22px;'>{label}</span></p>",
        unsafe_allow_html=True
    )

    # === PETA ===
    st.markdown("<div class='section-title'>Visualisasi Lokasi</div>", unsafe_allow_html=True)
    m = folium.Map(location=[-0.5071, 101.4478], zoom_start=11)
    color_map = {"Low": "blue", "Moderate": "green", "High": "orange", "Very High": "red"}
    popup_html = f"""
        <b>Prediksi:</b> {label}<br>
        <b>Suhu:</b> {last_row[fitur[0]]} °C<br>
        <b>Kelembapan:</b> {last_row[fitur[1]]} %<br>
        <b>Curah Hujan:</b> {last_row[fitur[2]]} mm<br>
        <b>Angin:</b> {last_row[fitur[3]]} m/s<br>
        <b>Kelembaban Tanah:</b> {last_row[fitur[4]]} %<br>
        <b>Waktu:</b> {last_row["Waktu"]}
    """
    folium.Marker(
        location=[-0.5071, 101.4478],
        popup=folium.Popup(popup_html, max_width=250),
        icon=folium.Icon(color=color_map.get(label, "gray"))
    ).add_to(m)
    folium_static(m, width=850, height=500)

else:
    st.warning("Data sensor tidak tersedia.")
