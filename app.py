# ============================================================

# 🌐 Aplikasi Web Analisis Sentimen MBG - Model LSTM (Streamlit + GIF)

# ============================================================

import streamlit as st
import pickle
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === 1️⃣ Load model & komponen ===

model = load_model("model_lstm_mbg.keras")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# === 2️⃣ Parameter penting ===

maxlen = 100

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
        st.image("https://media.tenor.com/q8i5TxkKX40AAAAd/prabowo-wowo-presiden-2045-mulyono.gif")

    elif result == "negatif":
        st.error(f"💢 Sentimen: **{result.upper()}** 😠")
        st.image("https://media.tenor.com/ftGx8H6InAIAAAAd/kabur-meme.gif")

    else:  # netral
        st.info(f"😐 Sentimen: **{result.upper()}**")
        st.image("https://media.tenor.com/Q3PsN2eobwUAAAAd/gibran-savage-debat-cawapres.gif")


st.markdown("---")
st.caption("Dibangun dengan TensorFlow LSTM, Tokenizer, dan Streamlit 🧠")
