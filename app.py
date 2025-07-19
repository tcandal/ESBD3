import streamlit as st
import joblib
import os

st.title("Classificador de sentimentos")

texto = st.text_input("Digite tweet:")

if os.path.exists("vectorizer.joblib") and os.path.exists("model.joblib"):
    model = joblib.load("model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    if st.button("Classificar"):
        if texto.strip():
            vetor = vectorizer.transform([texto])
            pred = model.predict(vetor)
            st.write(f"Sentimento: {pred}")
        else:
            st.warning("Por favor, insira um tweet para classificar.")
else:
    st.error("Modelo ou vetor não encontrados. Certifique-se de que o modelo foi treinado e salvo corretamente.")

