import streamlit as st
import requests

st.title("📰 Fake News Detection")

user_input = st.text_area("Enter News Article Text")

if st.button("Check Authenticity"):
    response = requests.post("http://127.0.0.1:5000/predict", json={"text": user_input})
    
    if response.status_code == 200:
        result = response.json()['prediction']
        st.success(f"The news article is classified as: *{result}*")
    else:
        st.error("Error in prediction")