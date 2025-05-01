# === SMART FIRE PREDICTION RHSEM-IoT - STREAMLIT APP ===
import streamlit as st
import pandas as pd
import joblib
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
from io import BytesIO
import folium
from streamlit_folium import folium_static

# === PAGE CONFIG (HANYA SEKALI SAJA) ===
st.set_page_config(page_title="Smart Fire Prediction RHSEM - IoT", layout="wide")

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
    .scrollable-table {
        overflow-x: auto;
    }
    </style>
""", unsafe_allow_html=True)

# === FUNGSI BANTUAN ===
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

# === LOAD DATA TANPA CACHE ===
@st.cache_data(ttl=10)
def load_data():
    url = "https://docs.google.com/spreadsheets/d/1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM/export?format=csv"
    return pd.read_csv(url)

df = load_data()

# === HEADER ===
st.markdown("<hr style='margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 9])
with col1:
    st.image("logo.png", width=170)
with col2:
    st.markdown("""
        <div style='margin-left: 20px;'>
            <h2 style='margin-bottom: 0px;'>Smart Fire Prediction RHSEM - IoT Model</h2>
            <p style='font-size: 16px; line-height: 1.5; margin-top: 8px;'>
                Sistem ini menggunakan Rotational Hybrid Stacking Ensemble Method (RHSEM) untuk memprediksi risiko kebakaran hutan secara real-time dengan tingkat akurasi tinggi.
                Data pengujian berasal dari perangkat IoT yang mengukur parameter lingkungan seperti suhu, kelembapan, curah hujan, kecepatan angin, dan kelembapan tanah.
            </p>
        </div>
    """, unsafe_allow_html=True)

# === LINK DATA CLOUD ===
col_btn = st.columns([10, 1])[1]
with col_btn:
    st.markdown(
        """
        <a href='https://docs.google.com/spreadsheets/d/1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM/edit?gid=0#gid=0' target='_blank'>
        <button style='padding: 6px 16px; background-color: #1f77b4; color: white; border: none; border-radius: 4px; cursor: pointer;'>Data Cloud</button>
        </a>""",
        unsafe_allow_html=True
    )

st.markdown("<hr style='margin-top: 10px; margin-bottom: 25px;'>", unsafe_allow_html=True)

# === REALTIME PREDICTION ===
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

    clean_df = df[fitur].copy()
    for col in fitur:
        clean_df[col] = clean_df[col].astype(str).str.replace(',', '.').astype(float).fillna(0)

    scaled_all = scaler.transform(clean_df)
    predictions = [convert_to_label(p) for p in model.predict(scaled_all)]
    df["Prediksi Kebakaran"] = predictions

    last_row = df.iloc[-1]
    waktu = pd.to_datetime(last_row['Waktu'])
    hari = convert_day_to_indonesian(waktu.strftime('%A'))
    bulan = convert_month_to_indonesian(waktu.strftime('%B'))
    tanggal = waktu.strftime(f'%d {bulan} %Y')
    risk_label = last_row["Prediksi Kebakaran"]
    font, bg = risk_styles.get(risk_label, ("black", "white"))

    sensor_df = pd.DataFrame({"Variabel": fitur, "Value": [f"{last_row[col]:.1f}" for col in fitur]})
    st.write("Data Sensor Realtime:")
    st.table(sensor_df)

    st.markdown(
        f"<p style='background-color:{bg}; color:{font}; padding:10px; border-radius:8px; font-weight:bold;'>"
        f"Pada hari {hari}, tanggal {tanggal}, lahan ini diprediksi memiliki tingkat resiko kebakaran: "
        f"<span style='text-decoration: underline; font-size: 22px;'>{risk_label}</span></p>",
        unsafe_allow_html=True
    )

    # === PETA PEKANBARU ===
    st.markdown("<div class='section-title'>Peta Lokasi Prediksi Kebakaran</div>", unsafe_allow_html=True)
    pekanbaru_coords = [-0.5071, 101.4478]
    color_map = {"Low": "blue", "Moderate": "green", "High": "orange", "Very High": "red"}
    marker_color = color_map.get(risk_label, "blue")

    m = folium.Map(location=pekanbaru_coords, zoom_start=10)
    popup_text = folium.Popup(f"""
        <div style='width: 230px; font-size: 13px; line-height: 1.5;'>
        <b>Prediksi:</b> {risk_label}<br>
        <b>Suhu:</b> {last_row[fitur[0]]} °C<br>
        <b>Kelembapan:</b> {last_row[fitur[1]]} %<br>
        <b>Curah Hujan:</b> {last_row[fitur[2]]} mm<br>
        <b>Kecepatan Angin:</b> {last_row[fitur[3]]} m/s<br>
        <b>Kelembaban Tanah:</b> {last_row[fitur[4]]} %<br>
        <b>Waktu:</b> {last_row['Waktu']}</div>", max_width=250)

    folium.Marker(location=pekanbaru_coords, popup=popup_text,
                  icon=folium.Icon(color=marker_color, icon="info-sign")).add_to(m)
    folium_static(m, width=850, height=500)

# === TABEL KLASIFIKASI ===
st.markdown("<div class='section-title'>Tabel Tingkat Resiko dan Intensitas Kebakaran</div>", unsafe_allow_html=True)
st.markdown("""
<div class="scrollable-table" style="margin-bottom: 25px;">
<table>
<thead><tr><th>Warna</th><th>Tingkat Resiko / Intensitas</th><th>Keterangan</th></tr></thead>
<tbody>
<tr style='background-color:blue; color:white;'><td>Blue</td><td>Low</td><td style='text-align:left;'>Tingkat resiko kebakaran rendah. Api mudah dikendalikan.</td></tr>
<tr style='background-color:green; color:white;'><td>Green</td><td>Moderate</td><td style='text-align:left;'>Tingkat resiko sedang. Api relatif mudah dikendalikan.</td></tr>
<tr style='background-color:yellow; color:black;'><td>Yellow</td><td>High</td><td style='text-align:left;'>Tingkat resiko tinggi. Api sulit dikendalikan.</td></tr>
<tr style='background-color:red; color:white;'><td>Red</td><td>Very High</td><td style='text-align:left;'>Tingkat resiko sangat tinggi. Api sangat sulit dikendalikan.</td></tr>
</tbody></table></div>
""", unsafe_allow_html=True)

# === TABEL DATA SENSOR ===
st.markdown("<div class='section-title'>Data Sensor Lengkap</div>", unsafe_allow_html=True)
st.dataframe(df[['Waktu'] + fitur + ['Prediksi Kebakaran']], use_container_width=True)

# === EXPORT TO XLSX ===
output = BytesIO()
with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
    df.to_excel(writer, index=False, sheet_name='DataSensor')
xlsx_data = output.getvalue()
st.download_button("📥 Download Hasil Prediksi Kebakaran sebagai XLSX", xlsx_data, "hasil_prediksi_kebakaran.xlsx")

# === FOOTER ===
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<div style='margin-top: 20px; background-color: black; padding: 10px 20px; border-radius: 10px; text-align: center; color: white;'>
    <p style='margin: 0; font-size: 30px; font-weight: bold;'>Smart Fire Prediction RHSEM - IoT Model</p>
    <p style='margin: 0; font-size: 13px;'>Dikembangkan oleh Mahasiswa Universitas Putera Indonesia YPTK Padang Tahun 2025</p>
</div>
""", unsafe_allow_html=True)
