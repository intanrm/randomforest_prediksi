import streamlit as st
import pandas as pd
from model_training import train_model
from data_processing import load_and_preprocess_data
import matplotlib.pyplot as plt
import seaborn as sns

def recommend_kb(model, label_encoders, scaler, input_data):
    """
    Fungsi untuk memberikan rekomendasi alat kontrasepsi.
    """
    # Encode categorical data
    for col, encoder in label_encoders.items():
        if col in input_data:
            input_data[col] = encoder.transform([input_data[col]])[0]
    
    # Convert to DataFrame for scaling
    input_df = pd.DataFrame([input_data])
    
    # Scale the data
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    # Decode prediction if necessary
    recommendation = label_encoders['recommendation'].inverse_transform(prediction)[0]
    
    return recommendation

# Streamlit app
st.title("Sistem Rekomendasi Alat Kontrasepsi")

try:
    # Load and preprocess data
    X_train, X_test, y_train, y_test, label_encoders, scaler = load_and_preprocess_data("data_kb.csv")
    
    # Train model and generate confusion matrix
    model, conf_matrix = train_model(X_train, y_train, X_test, y_test)
except Exception as e:
    st.error(f"Error memuat data atau model: {e}")
    st.stop()

# Input dari pengguna
usia = st.slider("Masukkan Usia", min_value=18, max_value=50, value=25, step=1)
jenis_kelamin = st.radio("Jenis Kelamin", ["Pria", "Wanita"])
jumlah_anak = st.number_input("Jumlah Anak", min_value=0, max_value=10, value=2, step=1)
pendidikan = st.selectbox("Pendidikan", ["SD", "SMP", "SMA", "Perguruan Tinggi"])
status_ekonomi = st.selectbox("Status Ekonomi", ["Rendah", "Menengah", "Tinggi"])
pekerjaan = st.selectbox("Pekerjaan", ["Ibu Rumah Tangga", "PNS", "Karyawan Swasta", "Buruh", "Wiraswasta"])
sumber_informasi = st.selectbox("Sumber Informasi", ["Tenaga Kesehatan", "Media", "Keluarga", "Teman"])

input_data = {
    'usia': usia,
    'jumlah_anak': jumlah_anak,
    'jenis_kelamin': "Wanita" if jenis_kelamin == "Wanita" else "Pria",
    'pendidikan': pendidikan,
    'status_ekonomi': status_ekonomi,
    'pekerjaan': pekerjaan,
    'sumber_informasi': sumber_informasi,
}

try:
    # Generate recommendation
    recommendation = recommend_kb(model, label_encoders, scaler, input_data)
    st.success(f"Rekomendasi KB: {recommendation}")
except Exception as e:
    st.error(f"Error dalam memberikan rekomendasi: {e}")
