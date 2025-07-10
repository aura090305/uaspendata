import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
# Streamlit App
st.title("Prediksi Diabetes")
st.write("Masukkan data pasien untuk memprediksi apakah pasien positif atau negatif diabetes.")

# Input pengguna
age = st.number_input("Usia", min_value=0, max_value=120, step=1)
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
polyuria = st.selectbox("Polyuria", ["Yes", "No"])
polydipsia = st.selectbox("Polydipsia", ["Yes", "No"])
sudden_weight_loss = st.selectbox("Sudden Weight Loss", ["Yes", "No"])
weakness = st.selectbox("Weakness", ["Yes", "No"])
polyphagia = st.selectbox("Polyphagia", ["Yes", "No"])
genital_thrush = st.selectbox("Genital Thrush", ["Yes", "No"])
visual_blurring = st.selectbox("Visual Blurring", ["Yes", "No"])
itching = st.selectbox("Itching", ["Yes", "No"])
irritability = st.selectbox("Irritability", ["Yes", "No"])
delayed_healing = st.selectbox("Delayed Healing", ["Yes", "No"])
partial_paresis = st.selectbox("Partial Paresis", ["Yes", "No"])
muscle_stiffness = st.selectbox("Muscle Stiffness", ["Yes", "No"])
alopecia = st.selectbox("Alopecia", ["Yes", "No"])
obesity = st.selectbox("Obesity", ["Yes", "No"])

# Tombol untuk prediksi
if st.button("Prediksi"):
    # Membuat DataFrame dari input pengguna
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'polyuria': [polyuria],
        'polydipsia': [polydipsia],
        'sudden_weight_loss': [sudden_weight_loss],
        'weakness': [weakness],
        'polyphagia': [polyphagia],
        'genital_thrush': [genital_thrush],
        'visual_blurring': [visual_blurring],
        'itching': [itching],
        'irritability': [irritability],
        'delayed_healing': [delayed_healing],
        'partial_paresis': [partial_paresis],
        'muscle_stiffness': [muscle_stiffness],
        'alopecia': [alopecia],
        'obesity': [obesity]
    })

        # Preprocessing input pengguna
    input_data['gender'] = LabelEncoder().fit_transform(input_data['gender'])
    yes_no_columns = ['polyuria', 'polydipsia', 'sudden_weight_loss', 'weakness', 'polyphagia',
                      'genital_thrush', 'visual_blurring', 'itching', 'irritability',
                      'delayed_healing', 'partial_paresis', 'muscle_stiffness', 'alopecia', 'obesity']
    for col in yes_no_columns:
        input_data[col] = input_data[col].map({'Yes': 1, 'No': 0})

    # Memuat model
    model = joblib.load('model_decision_tree.pkl')

    # Prediksi
    prediction = model.predict(input_data)
    prediction_prob = model.predict_proba(input_data)[0]

    # Menampilkan hasil
    result = "Positif Diabetes" if prediction[0] == 1 else "Negatif Diabetes"
    st.write(f"**Hasil Prediksi**: {result}")
    st.write(f"**Probabilitas Positif**: {prediction_prob[1]:.2%}")
    st.write(f"**Probabilitas Negatif**: {prediction_prob[0]:.2%}")