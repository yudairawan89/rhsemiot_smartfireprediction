import streamlit as st
st.set_page_config(page_title="RHSEM-IoT Smart Fire Prediction", page_icon="🔥")

import pandas as pd
import joblib
from PIL import Image
from streamlit_autorefresh import st_autorefresh

# === Fungsi Utilitas ===
def convert_day_to_indonesian(day_name):
    days = {
        'Monday': 'Senin', 'Tuesday': 'Selasa', 'Wednesday': 'Rabu',
        'Thursday': 'Kamis', 'Friday': 'Jumat', 'Saturday': 'Sabtu', 'Sunday': 'Minggu'
    }
    return days.get(day_name, day_name)

def convert_month_to_indonesian(month_name):
    months = {
        'January': 'Januari', 'February': 'Februari', 'March': 'Maret',
        'April': 'April', 'May': 'Mei', 'June': 'Juni',
        'July': 'Juli', 'August': 'Agustus', 'September': 'September',
        'October': 'Oktober', 'November': 'November', 'December': 'Desember'
    }
    return months.get(month_name, month_name)

def convert_to_label(pred):
    mapping = {
        0: "Low / Rendah",
        1: "Moderate / Sedang",
        2: "High / Tinggi",
        3: "Very High / Sangat Tinggi"
    }
    return mapping.get(pred, "Unknown")

risk_styles = {
    "Low / Rendah": {"color": "white", "background-color": "blue"},
    "Moderate / Sedang": {"color": "white", "background-color": "green"},
    "High / Tinggi": {"color": "black", "background-color": "yellow"},
    "Very High / Sangat Tinggi": {"color": "white", "background-color": "red"}
}

# === Auto Refresh Tiap 3 Detik ===
st_autorefresh(interval=3000, key="data_refresh")

# === Header dan Logo ===
col1, col2 = st.columns([1, 6])
with col1:
    st.image("logo.png", width=100)
with col2:
    st.title("UHTP Smart Fire Prediction")

st.markdown("""
Sistem Prediksi Tingkat Resiko Kebakaran Hutan dan Lahan berbasis RHSEM-IoT.  
Data diambil secara realtime dari perangkat sensor dan disimpan ke [Google Sheets](https://docs.google.com/spreadsheets/d/1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM).
""")

# === Load Data dan Model ===
@st.cache_data
def load_data():
    url = 'https://docs.google.com/spreadsheets/d/1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM/export?format=csv'
    return pd.read_csv(url)

@st.cache_resource
def load_model():
    return joblib.load("RHSEM_IoT_Model.joblib")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.joblib")

df = load_data()
model = load_model()
scaler = load_scaler()

# === Rename Kolom dan Prediksi Realtime ===
if df is not None:
    df = df.rename(columns={
        'Suhu Udara': 'Tavg: Temperatur rata-rata (°C)',
        'Kelembapan Udara': 'RH_avg: Kelembapan rata-rata (%)',
        'Curah Hujan/Jam': 'RR: Curah hujan (mm)',
        'Kecepatan Angin (ms)': 'ff_avg: Kecepatan angin rata-rata (m/s)',
        'Kelembapan Tanah': 'Kelembaban Permukaan Tanah',
        'Waktu': 'Waktu'
    })

    fitur = ['Tavg: Temperatur rata-rata (°C)', 'RH_avg: Kelembapan rata-rata (%)',
             'RR: Curah hujan (mm)', 'ff_avg: Kecepatan angin rata-rata (m/s)', 'Kelembaban Permukaan Tanah']

    if all(f in df.columns for f in fitur):
        # Bersihkan dan Prediksi
        fitur_data = df[fitur].astype(str).apply(lambda x: x.str.replace(',', '.')).astype(float).fillna(0)
        scaled = scaler.transform(fitur_data)
        df['Prediksi Kebakaran'] = [convert_to_label(p) for p in model.predict(scaled)]

        last_row = df.iloc[-1]
        waktu = pd.to_datetime(last_row['Waktu'])
        hari = convert_day_to_indonesian(waktu.strftime('%A'))
        bulan = convert_month_to_indonesian(waktu.strftime('%B'))
        tanggal = waktu.strftime(f'%d {bulan} %Y')
        label = last_row['Prediksi Kebakaran']
        style = risk_styles.get(label, {"color": "black", "background-color": "white"})

        # === Tampilkan Prediksi Terbaru ===
        st.subheader("📡 Prediksi Realtime")
        st.markdown(
            f"<div style='background-color:{style['background-color']}; color:{style['color']}; "
            f"padding: 10px; border-radius: 10px;'>"
            f"<b>Pada hari {hari}, {tanggal}, tingkat risiko kebakaran diprediksi:</b><br>"
            f"<h2 style='text-decoration:underline;'>{label}</h2></div>",
            unsafe_allow_html=True
        )

        # === Tabel Sensor Terbaru ===
        st.write("**📊 Data Sensor Realtime**")
        st.dataframe(last_row[fitur])

# === Prediksi Manual ===
st.subheader("📝 Prediksi Manual")
col1, col2 = st.columns(2)
with col1:
    suhu = st.number_input("Suhu Udara (°C)", value=30.0)
    kelembapan = st.number_input("Kelembapan Udara (%)", value=60.0)
    curah_hujan = st.number_input("Curah Hujan (mm)", value=10.0)
with col2:
    angin = st.number_input("Kecepatan Angin (m/s)", value=2.0)
    tanah = st.number_input("Kelembaban Tanah (%)", value=50.0)

if st.button("Prediksi Manual"):
    input_df = pd.DataFrame([{
        'Tavg: Temperatur rata-rata (°C)': suhu,
        'RH_avg: Kelembapan rata-rata (%)': kelembapan,
        'RR: Curah hujan (mm)': curah_hujan,
        'ff_avg: Kecepatan angin rata-rata (m/s)': angin,
        'Kelembaban Permukaan Tanah': tanah
    }])
    scaled = scaler.transform(input_df)
    pred = convert_to_label(model.predict(scaled)[0])
    style = risk_styles.get(pred, {"color": "black", "background-color": "white"})

    st.markdown(
        f"<p style='color:{style['color']}; background-color:{style['background-color']}; "
        f"font-weight:bold; padding:12px; border-radius:6px;'>Prediksi Manual: {pred}</p>",
        unsafe_allow_html=True
    )

# === Footer ===
st.markdown("---")
col1, col2 = st.columns([1, 3])
with col1:
    st.image("upi.png", width=600)
with col2:
    st.markdown("""
        <div style='text-align: center;'>
            <h4>Smart Fire Prediction RHSEM – IoT</h4>
            <p>Dikembangkan oleh Mahasiswa Universitas Putera Indonesia YPTK Padang, 2025</p>
        </div>
    """, unsafe_allow_html=True)
