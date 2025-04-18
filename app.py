import streamlit as st
import pandas as pd
import joblib
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
from io import BytesIO

st.set_page_config(page_title="Smart Fire Prediction RHSEM – IoT", layout="wide")

# === AUTO REFRESH ===
st_autorefresh(interval=3000, key="data_refresh")

# === STYLE KUSTOM ===
st.markdown("""
    <style>
    .main {background-color: #F9F9F9;}
    .section-title {
        background-color: #1f77b4;
        color: white;
        padding: 10px;
        border-radius: 6px;
        font-weight: bold;
    }
    .scroll-table {
        overflow-x: auto;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 1em;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 8px;
    }
    th {
        background-color: #e0e0e0;
        text-align: center;
    }
    td {
        text-align: center;
    }
    td.keterangan {
        text-align: left !important;
    }
    </style>
""", unsafe_allow_html=True)

# === DICTIONARY & STYLE ===
def convert_day_to_indonesian(day):
    return {'Monday': 'Senin', 'Tuesday': 'Selasa', 'Wednesday': 'Rabu',
            'Thursday': 'Kamis', 'Friday': 'Jumat', 'Saturday': 'Sabtu',
            'Sunday': 'Minggu'}.get(day, day)

def convert_month_to_indonesian(month):
    return {'January': 'Januari', 'February': 'Februari', 'March': 'Maret',
            'April': 'April', 'May': 'Mei', 'June': 'Juni', 'July': 'Juli',
            'August': 'Agustus', 'September': 'September', 'October': 'Oktober',
            'November': 'November', 'December': 'Desember'}.get(month, month)

def convert_to_label(pred):
    return {0: "Low", 1: "Moderate", 2: "High", 3: "Very High"}.get(pred, "Unknown")

risk_styles = {
    "Low": ("white", "blue"),
    "Moderate": ("white", "green"),
    "High": ("black", "yellow"),
    "Very High": ("white", "red")
}

# === LOAD MODEL & SCALER ===
@st.cache_resource
def load_model():
    return joblib.load("RHSEM_IoT_Model.joblib")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.joblib")

model = load_model()
scaler = load_scaler()

# === LOAD DATA TANPA CACHE ===
def load_data():
    url = "https://docs.google.com/spreadsheets/d/1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM/export?format=csv"
    return pd.read_csv(url)

st.cache_data.clear()
df = load_data()

# === HEADER & DESKRIPSI ===
col1, col2 = st.columns([1, 9])
with col1:
    st.image("logo.png", width=90)
with col2:
    st.markdown("<h2 style='margin-bottom:0;'>Smart Fire Prediction RHSEM – IoT Model</h2>", unsafe_allow_html=True)
    subcol, btncol = st.columns([8, 2])
    with subcol:
        st.caption("Sistem Prediksi Tingkat Resiko Kebakaran Hutan dan Lahan menggunakan model Hybrid Machine & Deep Learning. Data diambil dari IoT secara Realtime via Google Sheets.")
    with btncol:
        st.markdown(
            "<a href='https://docs.google.com/spreadsheets/d/1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM/edit?gid=0#gid=0' target='_blank'>"
            "<button style='padding: 6px 16px; background-color: #1f77b4; color: white; border: none; border-radius: 4px; cursor: pointer;'>Data Cloud</button>"
            "</a>",
            unsafe_allow_html=True
        )

# === PROSES & PREDIKSI ===
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

    # Prediksi untuk semua baris kosong
    for i, row in df.iterrows():
        try:
            if pd.isna(row.get("Prediksi Kebakaran", None)):
                clean = [float(str(row[col]).replace(',', '.')) if col in row else 0 for col in fitur]
                scaled = scaler.transform([clean])
                df.at[i, "Prediksi Kebakaran"] = convert_to_label(model.predict(scaled)[0])
        except:
            df.at[i, "Prediksi Kebakaran"] = "Unknown"

    last_row = df.iloc[-1]
    waktu = pd.to_datetime(last_row['Waktu'])
    hari = convert_day_to_indonesian(waktu.strftime('%A'))
    bulan = convert_month_to_indonesian(waktu.strftime('%B'))
    tanggal = waktu.strftime(f'%d {bulan} %Y')
    risk = last_row["Prediksi Kebakaran"]
    font, bg = risk_styles.get(risk, ("black", "white"))

    # Tampilan sensor terakhir
    st.write("Data Sensor Realtime:")
    sensor_df = pd.DataFrame({
        "Variabel": fitur,
        "Value": [last_row[col] for col in fitur]
    })
    st.table(sensor_df)

    # Tampilan hasil
    st.markdown(
        f"<p style='background-color:{bg}; color:{font}; padding:10px; border-radius:8px; font-weight:bold;'>"
        f"Pada hari {hari}, tanggal {tanggal}, lahan ini diprediksi memiliki tingkat resiko kebakaran: "
        f"<span style='text-decoration: underline; font-size: 22px;'>{risk}</span></p>",
        unsafe_allow_html=True
    )

# === TABEL TINGKAT RESIKO ===
st.markdown("<div class='section-title'>Tabel Tingkat Resiko dan Intensitas Kebakaran</div>", unsafe_allow_html=True)
st.markdown("""
<div class='scroll-table'>
<table>
    <thead>
        <tr>
            <th style='width:10%; background-color:blue; color:white;'>Blue</th>
            <th style='width:15%'>Low</th>
            <td class='keterangan'>Resiko rendah, api mudah dikendalikan dan padam sendiri.</td>
        </tr>
        <tr>
            <th style='background-color:green; color:white;'>Green</th>
            <th>Moderate</th>
            <td class='keterangan'>Resiko sedang, api relatif masih dapat dikendalikan.</td>
        </tr>
        <tr>
            <th style='background-color:yellow;'>Yellow</th>
            <th>High</th>
            <td class='keterangan'>Resiko tinggi, api mulai sulit dikendalikan.</td>
        </tr>
        <tr>
            <th style='background-color:red; color:white;'>Red</th>
            <th>Very High</th>
            <td class='keterangan'>Resiko sangat tinggi, api sangat sulit dikendalikan.</td>
        </tr>
    </thead>
</table>
</div>
""", unsafe_allow_html=True)

# === TABEL DATA SENSOR LENGKAP ===
st.markdown("<div class='section-title'>Data Sensor Lengkap</div>", unsafe_allow_html=True)
st.dataframe(df[['Waktu'] + fitur + ['Prediksi Kebakaran']], use_container_width=True)

# === UNDUH DATA ===
output = BytesIO()
with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
    df.to_excel(writer, index=False, sheet_name='DataSensor')
st.download_button(
    "📥 Download Hasil Prediksi Kebakaran sebagai XLSX",
    data=output.getvalue(),
    file_name='hasil_prediksi_kebakaran.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)

# === PREDIKSI MANUAL ===
st.markdown("<div class='section-title'>Pengujian Menggunakan Data Manual</div>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    suhu = st.number_input("Suhu Udara (°C)", value=st.session_state.get("suhu", 30.0), key="suhu")
    kelembapan = st.number_input("Kelembapan Udara (%)", value=st.session_state.get("lembab", 65.0), key="lembab")
with col2:
    curah = st.number_input("Curah Hujan (mm)", value=st.session_state.get("curah", 10.0), key="curah")
    angin = st.number_input("Kecepatan Angin (m/s)", value=st.session_state.get("angin", 3.0), key="angin")
with col3:
    tanah = st.number_input("Kelembaban Tanah (%)", value=st.session_state.get("tanah", 50.0), key="tanah")

if st.button("Prediksi Manual"):
    input_df = pd.DataFrame([{
        'Tavg: Temperatur rata-rata (°C)': suhu,
        'RH_avg: Kelembapan rata-rata (%)': kelembapan,
        'RR: Curah hujan (mm)': curah,
        'ff_avg: Kecepatan angin rata-rata (m/s)': angin,
        'Kelembaban Permukaan Tanah': tanah
    }])
    scaled_manual = scaler.transform(input_df)
    hasil = convert_to_label(model.predict(scaled_manual)[0])
    font, bg = risk_styles.get(hasil, ("black", "white"))
    st.markdown(
        f"<p style='color:{font}; background-color:{bg}; padding:10px; border-radius:5px;'>"
        f"Prediksi Risiko Kebakaran: <b>{hasil}</b></p>", unsafe_allow_html=True
    )

# === FOOTER ===
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<center>
<h4>Smart Fire Prediction RHSEM – IoT Model</h4>
<p>Dikembangkan oleh Mahasiswa Universitas Putera Indonesia YPTK Padang Tahun 2025</p>
</center>
""", unsafe_allow_html=True)
