# ============================================================

# 🌐 Aplikasi Web Analisis Sentimen MBG - Model LSTM (Streamlit)

# ============================================================

import streamlit as st
import pickle
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === 1️⃣ Load model & komponen ===

# Pastikan nama file sesuai dengan yang kamu upload di repo

model = load_model("model_lstm_mbg.keras")  # atau model_lstm_mbg.h5 jika kamu ubah
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))  # isinya Tokenizer
le = pickle.load(open("label_encoder.pkl", "rb"))

# === 2️⃣ Parameter penting ===

maxlen = 100  # harus sama seperti saat training

# === 3️⃣ Fungsi pembersihan teks ===

def clean_text(text):
text = str(text).lower()
text = re.sub(r"http\S+|www\S+", "", text)
text = re.sub(r"@\w+", "", text)
text = re.sub(r"#\w+", "", text)
text = re.sub(r"[^a-zA-Z\s]", "", text)
text = re.sub(r"\s+", " ", text).strip()
return text

# === 4️⃣ Fungsi prediksi ===

def predict_sentiment(text):
clean = clean_text(text)
seq = tokenizer.texts_to_sequences([clean])
padded = pad_sequences(seq, maxlen=maxlen, padding='post', truncating='post')
pred = model.predict(padded)
label = le.inverse_transform([np.argmax(pred)])
return label[0]

# === 5️⃣ UI Streamlit ===

st.title("📘 Analisis Sentimen MBG dengan LSTM")
st.write("Masukkan teks untuk mengetahui sentimennya (positif, negatif, atau netral).")

user_input = st.text_area("📝 Masukkan teks:", "")

if st.button("Prediksi Sentimen"):
if user_input.strip() == "":
st.warning("Teks tidak boleh kosong.")
else:
result = predict_sentiment(user_input)
if result == "positif":
st.success(f"✨ Sentimen: **{result.upper()}** 😊")
st.image("[https://tenor.com/en-GB/view/prabowo-wowo-presiden-2045-mulyono-gif-9284220189663362549](https://tenor.com/en-GB/view/prabowo-wowo-presiden-2045-mulyono-gif-9284220189663362549)")
elif result == "negatif":
st.error(f"💢 Sentimen: **{result.upper()}** 😠")
st.image("[https://tenor.com/en-GB/view/kabur-meme-gif-15154939118701081091](https://tenor.com/en-GB/view/kabur-meme-gif-15154939118701081091)")
else:
st.info(f"😐 Sentimen: **{result.upper()}**")
st.image("[https://tenor.com/en-GB/view/gibran-savage-debat-cawapres-gif-5542767369797174458](https://tenor.com/en-GB/view/gibran-savage-debat-cawapres-gif-5542767369797174458)")

st.markdown("---")
st.caption("Dibangun dengan TensorFlow LSTM, Tokenizer, dan Streamlit 🧠")
