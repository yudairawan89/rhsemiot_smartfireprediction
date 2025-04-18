# === Smart Fire Prediction RHSEM – IoT ===

import streamlit as st
import pandas as pd
import joblib
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
from io import BytesIO

# === Konfigurasi halaman ===
st.set_page_config(page_title="Smart Fire Prediction RHSEM – IoT", layout="wide")

# === Auto refresh setiap 3 detik ===
st_autorefresh(interval=3000, key="data_refresh")

# === Gaya visual CSS ===
st.markdown("""
<style>
.section-title {
    background-color: #1f77b4;
    color: white;
    padding: 10px;
    border-radius: 6px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# === Fungsi utilitas ===
def convert_day_to_indonesian(day_name):
    return {
        'Monday': 'Senin', 'Tuesday': 'Selasa', 'Wednesday': 'Rabu',
        'Thursday': 'Kamis', 'Friday': 'Jumat', 'Saturday': 'Sabtu',
        'Sunday': 'Minggu'
    }.get(day_name, day_name)

def convert_month_to_indonesian(month_name):
    return {
        'January': 'Januari', 'February': 'Februari', 'March': 'Maret',
        'April': 'April', 'May': 'Mei', 'June': 'Juni', 'July': 'Juli',
        'August': 'Agustus', 'September': 'September', 'October': 'Oktober',
        'November': 'November', 'December': 'Desember'
    }.get(month_name, month_name)

def convert_to_label(pred):
    return {0: "Low", 1: "Moderate", 2: "High", 3: "Very High"}.get(pred, "Unknown")

risk_styles = {
    "Low": ("white", "blue"),
    "Moderate": ("white", "green"),
    "High": ("black", "yellow"),
    "Very High": ("white", "red")
}

# === Load model dan scaler ===
@st.cache_resource
def load_model():
    return joblib.load("RHSEM_IoT_Model.joblib")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.joblib")

model = load_model()
scaler = load_scaler()

# === Load data realtime ===
@st.cache_data
def load_data():
    url = "https://docs.google.com/spreadsheets/d/1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM/export?format=csv"
    return pd.read_csv(url)

st.cache_data.clear()
df = load_data()

# === Header Aplikasi ===
col1, col2 = st.columns([1, 9])
with col1:
    st.image("logo.png", width=90)
with col2:
    st.markdown("<h2 style='margin-bottom:0;'>Smart Fire Prediction RHSEM – IoT Model</h2>", unsafe_allow_html=True)
    c1, c2 = st.columns([8, 2])
    with c1:
        st.caption("Sistem Prediksi Tingkat Resiko Kebakaran Hutan dan Lahan menggunakan model Hybrid Machine & Deep Learning. Data diambil dari IoT secara Realtime via Google Sheets.")
    with c2:
        st.markdown(
            "<a href='https://docs.google.com/spreadsheets/d/1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM/edit?gid=0#gid=0' target='_blank'>"
            "<button style='padding: 6px 16px; background-color: #1f77b4; color: white; border: none; border-radius: 4px;'>Data Cloud</button>"
            "</a>",
            unsafe_allow_html=True
        )

# === Prediksi Real-Time untuk Semua Baris ===
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

    # Preprocessing untuk seluruh baris
    X_clean = df[fitur].copy()
    for f in fitur:
        X_clean[f] = X_clean[f].astype(str).str.replace(',', '.').astype(float)
    X_clean = X_clean.fillna(0)

    # Prediksi semua baris
    X_scaled = scaler.transform(X_clean)
    df['Prediksi Kebakaran'] = [convert_to_label(l) for l in model.predict(X_scaled)]

    # Ambil baris terakhir untuk ditampilkan
    last_row = df.iloc[-1]
    waktu = pd.to_datetime(last_row['Waktu'])
    hari = convert_day_to_indonesian(waktu.strftime('%A'))
    bulan = convert_month_to_indonesian(waktu.strftime('%B'))
    tanggal = waktu.strftime(f'%d {bulan} %Y')
    label = last_row['Prediksi Kebakaran']
    font, bg = risk_styles.get(label, ("black", "white"))

    st.write("Data Sensor Realtime:")
    st.table(pd.DataFrame({
        "Variabel": fitur,
        "Value": [f"{last_row[f]:.1f}" for f in fitur]
    }))

    st.markdown(
        f"<p style='background-color:{bg}; color:{font}; padding:10px; border-radius:8px; font-weight:bold;'>"
        f"Pada hari {hari}, tanggal {tanggal}, lahan ini diprediksi memiliki tingkat resiko kebakaran: "
        f"<span style='text-decoration: underline; font-size: 22px;'>{label}</span></p>",
        unsafe_allow_html=True
    )

# === Tabel Risiko dan Keterangan ===
st.markdown("<div class='section-title'>Tabel Tingkat Resiko dan Intensitas Kebakaran</div>", unsafe_allow_html=True)
st.markdown("""
<table>
<thead>
<tr><th style='background-color:blue; color:white;'>Blue</th><th>Low</th><td>Resiko rendah, api mudah dikendalikan dan padam sendiri.</td></tr>
<tr><th style='background-color:green; color:white;'>Green</th><th>Moderate</th><td>Resiko sedang, api relatif masih dapat dikendalikan.</td></tr>
<tr><th style='background-color:yellow;'>Yellow</th><th>High</th><td>Resiko tinggi, api mulai sulit dikendalikan.</td></tr>
<tr><th style='background-color:red; color:white;'>Red</th><th>Very High</th><td>Resiko sangat tinggi, api sangat sulit dikendalikan.</td></tr>
</thead>
</table>
""", unsafe_allow_html=True)

# === Tabel Data Sensor Lengkap dengan Scroll ===
st.markdown("<div class='section-title'>Data Sensor Lengkap</div>", unsafe_allow_html=True)
st.dataframe(df[['Waktu'] + fitur + ['Prediksi Kebakaran']], height=400, use_container_width=True)

# === Export ke XLSX ===
output = BytesIO()
with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
    df.to_excel(writer, index=False, sheet_name='DataSensor')
st.download_button(
    label="📥 Download Hasil Prediksi Kebakaran sebagai XLSX",
    data=output.getvalue(),
    file_name='hasil_prediksi_kebakaran.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)

# === Pengujian Manual ===
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
    input_manual = pd.DataFrame([{
        'Tavg: Temperatur rata-rata (°C)': suhu,
        'RH_avg: Kelembapan rata-rata (%)': kelembapan,
        'RR: Curah hujan (mm)': curah,
        'ff_avg: Kecepatan angin rata-rata (m/s)': angin,
        'Kelembaban Permukaan Tanah': tanah
    }])
    scaled_input = scaler.transform(input_manual)
    pred_label = convert_to_label(model.predict(scaled_input)[0])
    font, bg = risk_styles.get(pred_label, ("black", "white"))
    st.markdown(
        f"<p style='color:{font}; background-color:{bg}; padding:10px; border-radius:5px;'>"
        f"Prediksi Risiko Kebakaran: <b>{pred_label}</b></p>",
        unsafe_allow_html=True
    )

# === Footer ===
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<center>
<h4>Smart Fire Prediction RHSEM – IoT Model</h4>
<p>Dikembangkan oleh Mahasiswa Universitas Putera Indonesia YPTK Padang Tahun 2025</p>
</center>
""", unsafe_allow_html=True)
