import streamlit as st
import pandas as pd
import joblib
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
from io import BytesIO
from streamlit_folium import folium_static
import folium

# === PAGE CONFIG ===
st.set_page_config(page_title="Smart Fire Prediction RHSEM - IoT", layout="wide")

# === LOAD MODEL DAN SCALER ===
@st.cache_resource
def load_model():
    return joblib.load("RHSEM_IoT_Model.joblib")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.joblib")

model = load_model()
scaler = load_scaler()

# === LOAD DATA FUNCTION ===
def load_data():
    url = "https://docs.google.com/spreadsheets/d/1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM/export?format=csv"
    return pd.read_csv(url)

# === HELPER FUNCTIONS ===
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

# === HEADER ===
col1, col2 = st.columns([1, 9])
with col1:
    st.image("logo.png", width=170)
with col2:
    st.markdown("""
        <div style='margin-left: 20px;'>
            <h2 style='margin-bottom: 0px;'>Smart Fire Prediction RHSEM - IoT Model</h2>
            <p style='font-size: 16px; line-height: 1.5; margin-top: 8px;'>
                Sistem ini menggunakan Rotational Hybrid Stacking Ensemble Method (RHSEM) untuk memprediksi risiko kebakaran hutan secara real-time dengan tingkat akurasi tinggi.
                Data berasal dari perangkat IoT yang mengukur parameter lingkungan seperti suhu, kelembapan, curah hujan, kecepatan angin, dan kelembapan tanah.
            </p>
        </div>
    """, unsafe_allow_html=True)
    with st.columns([10, 1])[1]:
        st.markdown("""
            <a href='https://docs.google.com/spreadsheets/d/1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM/edit?gid=0' target='_blank'>
            <button style='padding: 6px 16px; background-color: #1f77b4; color: white; border: none; border-radius: 4px; cursor: pointer;'>Data Cloud</button>
            </a>
        """, unsafe_allow_html=True)

st.markdown("<hr style='margin-top: 10px; margin-bottom: 25px;'>", unsafe_allow_html=True)

# === PREDIKSI REALTIME DENGAN REFRESH OTOMATIS ===
with st.container():
    st_autorefresh(interval=3000, key="realtime_refresh")
    df = load_data()

    st.markdown("<div class='section-title'>Hasil Prediksi Data Realtime</div>", unsafe_allow_html=True)

    if not df.empty:
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

        scaled = scaler.transform(clean_df)
        df["Prediksi Kebakaran"] = [convert_to_label(p) for p in model.predict(scaled)]

        last_row = df.iloc[-1]
        waktu = pd.to_datetime(last_row['Waktu'])
        tanggal = waktu.strftime(f"%d {convert_month_to_indonesian(waktu.strftime('%B'))} %Y")
        hari = convert_day_to_indonesian(waktu.strftime('%A'))
        label = last_row["Prediksi Kebakaran"]
        font, bg = risk_styles[label]

        col_kiri, col_kanan = st.columns([1.2, 1.8])
        with col_kiri:
            st.markdown("**Data Sensor Realtime:**")
            st.table(pd.DataFrame({"Variabel": fitur, "Value": [f"{last_row[col]:.1f}" for col in fitur]}))
            st.markdown(
                f"<p style='background-color:{bg}; color:{font}; padding:10px; border-radius:8px; font-weight:bold;'>"
                f"Pada hari {hari}, tanggal {tanggal}, risiko kebakaran: <span style='font-size: 22px;'>{label}</span></p>",
                unsafe_allow_html=True
            )
        with col_kanan:
            st.markdown("**Visualisasi Lokasi:**")
            peta = folium.Map(location=[-0.5071, 101.4478], zoom_start=11)
            folium.Circle(location=[-0.5071, 101.4478], radius=3000,
                          color=bg, fill=True, fill_opacity=0.3).add_to(peta)
            folium.Marker(location=[-0.5071, 101.4478],
                          popup=f"Prediksi: {label}",
                          icon=folium.Icon(color=bg)).add_to(peta)
            folium_static(peta, width=450, height=340)

# === TABEL TINGKAT RISIKO ===
st.markdown("<div class='section-title'>Tabel Tingkat Resiko dan Intensitas Kebakaran</div>", unsafe_allow_html=True)
st.markdown("""
<div class="scrollable-table" style="margin-bottom: 25px;">
<table>
<thead>
<tr><th>Warna</th><th>Tingkat Resiko</th><th>Keterangan</th></tr>
</thead><tbody>
<tr style='background-color:blue; color:white;'>
<td>Blue</td><td>Low</td><td style='text-align:left;'>Risiko rendah. Api mudah dikendalikan dan bisa padam sendiri.</td>
</tr>
<tr style='background-color:green; color:white;'>
<td>Green</td><td>Moderate</td><td style='text-align:left;'>Risiko sedang. Masih bisa dikendalikan.</td>
</tr>
<tr style='background-color:yellow; color:black;'>
<td>Yellow</td><td>High</td><td style='text-align:left;'>Risiko tinggi. Api sulit dikendalikan.</td>
</tr>
<tr style='background-color:red; color:white;'>
<td>Red</td><td>Very High</td><td style='text-align:left;'>Risiko sangat tinggi. Api sangat sulit dikendalikan.</td>
</tr>
</tbody></table></div>
""", unsafe_allow_html=True)

# === DATAFRAME LENGKAP + EXPORT ===
st.markdown("<div class='section-title' style='margin-top: 30px;'>Data Sensor Lengkap</div>", unsafe_allow_html=True)
st.dataframe(df[['Waktu'] + fitur + ['Prediksi Kebakaran']], use_container_width=True)

output = BytesIO()
with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name='DataSensor', index=False)
st.download_button("📥 Download Hasil Prediksi Kebakaran", output.getvalue(),
                   file_name="hasil_prediksi_kebakaran.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# === PREDIKSI MANUAL ===
st.markdown("<div class='section-title'>Pengujian Menggunakan Data Meteorologi Manual</div>", unsafe_allow_html=True)
if "manual_input" not in st.session_state:
    st.session_state.manual_input = {"suhu": 30, "kelembapan": 70, "curah": 5, "angin": 2, "tanah": 50}
if "manual_result" not in st.session_state:
    st.session_state.manual_result = None

col1, col2, col3 = st.columns(3)
with col1:
    suhu = st.number_input("Suhu (°C)", value=st.session_state.manual_input["suhu"])
    kelembapan = st.number_input("Kelembapan Udara (%)", value=st.session_state.manual_input["kelembapan"])
with col2:
    curah = st.number_input("Curah Hujan (mm)", value=st.session_state.manual_input["curah"])
    angin = st.number_input("Kecepatan Angin (m/s)", value=st.session_state.manual_input["angin"])
with col3:
    tanah = st.number_input("Kelembaban Tanah (%)", value=st.session_state.manual_input["tanah"])

pred_btn, reset_btn, _ = st.columns([1, 1, 8])
with pred_btn:
    if st.button("🔍 Prediksi Manual"):
        input_df = pd.DataFrame([{
            'Tavg: Temperatur rata-rata (°C)': suhu,
            'RH_avg: Kelembapan rata-rata (%)': kelembapan,
            'RR: Curah hujan (mm)': curah,
            'ff_avg: Kecepatan angin rata-rata (m/s)': angin,
            'Kelembaban Permukaan Tanah': tanah
        }])
        scaled_manual = scaler.transform(input_df)
        label = convert_to_label(model.predict(scaled_manual)[0])
        st.session_state.manual_result = label
        st.session_state.manual_input.update({"suhu": suhu, "kelembapan": kelembapan,
                                               "curah": curah, "angin": angin, "tanah": tanah})

with reset_btn:
    if st.button("🧼 Reset Manual"):
        st.session_state.manual_input = {"suhu": 0, "kelembapan": 0, "curah": 0, "angin": 0, "tanah": 0}
        st.session_state.manual_result = None
        st.experimental_rerun()

if st.session_state.manual_result:
    label = st.session_state.manual_result
    font, bg = risk_styles[label]
    st.markdown(f"<p style='color:{font}; background-color:{bg}; padding:10px; border-radius:6px;'>"
                f"Hasil Prediksi Manual: <b>{label}</b></p>", unsafe_allow_html=True)

# === PREDIKSI BERDASARKAN TEKS ===
st.markdown("<div class='section-title'>Pengujian Menggunakan Data Teks</div>", unsafe_allow_html=True)
if "text_input" not in st.session_state:
    st.session_state.text_input = ""
if "text_result" not in st.session_state:
    st.session_state.text_result = None

input_text = st.text_area("Masukkan deskripsi lingkungan (misal: 'cuaca panas dan tanah kering')",
                          value=st.session_state.text_input, height=120)

col_text_pred, col_text_reset, _ = st.columns([1, 1, 8])
with col_text_pred:
    if st.button("🔍 Prediksi Teks"):
        try:
            vectorizer = joblib.load("tfidf_vectorizer2.joblib")
            model_text = joblib.load("stacking_text_model2.joblib")
            X_vec = vectorizer.transform([input_text])
            pred = model_text.predict(X_vec)[0]
            label = convert_to_label(pred)
            st.session_state.text_input = input_text
            st.session_state.text_result = label
        except Exception as e:
            st.error(f"Terjadi error saat prediksi teks: {e}")

with col_text_reset:
    if st.button("🧼 Reset Teks"):
        st.session_state.text_input = ""
        st.session_state.text_result = None
        st.experimental_rerun()

if st.session_state.text_result:
    label = st.session_state.text_result
    font, bg = risk_styles[label]
    st.markdown(f"<p style='color:{font}; background-color:{bg}; padding:10px; border-radius:6px;'>"
                f"Hasil Prediksi Teks: <b>{label}</b></p>", unsafe_allow_html=True)

# === FOOTER ===
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<div style='margin-top: 20px; background-color: black; padding: 10px 20px; border-radius: 10px; text-align: center; color: white;'>
    <p style='margin: 0; font-size: 30px; font-weight: bold;'>Smart Fire Prediction RHSEM – IoT Model</p>
    <p style='margin: 0; font-size: 13px;'>Dikembangkan oleh Mahasiswa Universitas Putera Indonesia YPTK Padang Tahun 2025</p>
</div>
""", unsafe_allow_html=True)
