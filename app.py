import streamlit as st
import pandas as pd
from model_training import train_model
from data_processing import load_and_preprocess_data
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Sistem Rekomendasi Alat Kontrasepsi")

try:
    X_train, X_test, y_train, y_test, feature_mapping = load_and_preprocess_data("data_kb.csv")
    model, conf_matrix = train_model(X_train, y_train, X_test, y_test)
except Exception as e:
    st.error(f"Error memuat data atau model: {e}")
    st.stop()


expected_features = X_train.columns.tolist()


usia = st.slider("Masukkan Usia", min_value=18, max_value=50, value=25, step=1)
jenis_kelamin = st.radio("Jenis Kelamin", ["Pria", "Wanita"])
jumlah_anak = st.number_input("Jumlah Anak", min_value=0, max_value=10, value=2, step=1)
pendidikan = st.selectbox("Pendidikan", ["SD", "SMP", "SMA", "Perguruan Tinggi"])
status_ekonomi = st.selectbox("Status Ekonomi", ["Rendah", "Menengah", "Tinggi"])
pekerjaan = st.selectbox("Pekerjaan", ["Ibu Rumah Tangga", "PNS", "Karyawan Swasta", "Buruh", "Wiraswasta"])
sumber_informasi = st.selectbox("Sumber Informasi", ["Tenaga Kesehatan", "Media", "Keluarga", "Teman"])


input_data = pd.DataFrame({
    "Usia": [usia],
    "Jenis_Kelamin_Wanita": [1 if jenis_kelamin == "Wanita" else 0],
    "Jumlah_Anak": [jumlah_anak],
    "Pendidikan_SMP": [1 if pendidikan == "SMP" else 0],
    "Pendidikan_SMA": [1 if pendidikan == "SMA" else 0],
    "Pendidikan_Perguruan Tinggi": [1 if pendidikan == "Perguruan Tinggi" else 0],
    "Status_Ekonomi_Menengah": [1 if status_ekonomi == "Menengah" else 0],
    "Status_Ekonomi_Tinggi": [1 if status_ekonomi == "Tinggi" else 0],
    "Pekerjaan_Karyawan Swasta": [1 if pekerjaan == "Karyawan Swasta" else 0],
    "Pekerjaan_PNS": [1 if pekerjaan == "PNS" else 0],
    "Pekerjaan_Wiraswasta": [1 if pekerjaan == "Wiraswasta" else 0],
    "Sumber_Informasi_Media": [1 if sumber_informasi == "Media" else 0],
    "Sumber_Informasi_Teman": [1 if sumber_informasi == "Teman" else 0],
    "Sumber_Informasi_Keluarga": [1 if sumber_informasi == "Keluarga" else 0]
})

for col in expected_features:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[expected_features]


st.subheader("Data yang Digunakan untuk Prediksi")
st.write(input_data)

if st.button("Dapatkan Rekomendasi"):
    try:
        prediction = model.predict(input_data)[0]
        rekomendasi = ""
        if prediction == 1:
            rekomendasi = "Rekomendasi: Pilih alat kontrasepsi hormonal seperti pil KB atau suntik KB."
            st.success(rekomendasi)
        else:
            rekomendasi = "Rekomendasi: Pilih alat kontrasepsi non-hormonal seperti IUD atau kondom."
            st.warning(rekomendasi)
    except Exception as e:
        st.error(f"Error saat prediksi: {e}")

st.subheader("Confusion Matrix")
try:
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
except Exception as e:
    st.error(f"Error saat menampilkan confusion matrix: {e}")
