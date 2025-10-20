# ============================================================
# 🌐 Aplikasi Web Analisis Sentimen MBG - Versi Streamlit
# ============================================================

import streamlit as st
import pickle
import re
import numpy as np
from tensorflow.keras.models import load_model

# === 1️⃣ Load model & komponen ===
model = load_model("model_lstm_mbg.h5")
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# === 2️⃣ Fungsi pembersihan teks ===
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\\S+|www\\S+", "", text)
    text = re.sub(r"@\\w+", "", text)
    text = re.sub(r"#\\w+", "", text)
    text = re.sub(r"[^a-zA-Z\\s]", "", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text

# === 3️⃣ Fungsi prediksi ===
def predict_sentiment(text):
    clean = clean_text(text)
    vec = tfidf.transform([clean]).toarray()
    pred = model.predict(vec)
    label = le.inverse_transform([np.argmax(pred)])
    return label[0]

# === 4️⃣ UI Streamlit ===
st.title("📘 Analisis Sentimen MBG dengan Jaringan Syaraf Tiruan")
st.write("Masukkan teks untuk mengetahui sentimennya (positif, negatif, netral).")

user_input = st.text_area("📝 Masukkan teks:", "")
if st.button("Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("Teks tidak boleh kosong.")
    else:
        result = predict_sentiment(user_input)
        if result == "positif":
            st.success(f"✨ Sentimen: **{result.upper()}** 😊")
        elif result == "negatif":
            st.error(f"💢 Sentimen: **{result.upper()}** 😠")
        else:
            st.info(f"😐 Sentimen: **{result.upper()}**")

st.markdown("---")
st.caption("Dibangun dengan TensorFlow, TF-IDF, dan Streamlit 🧠")
