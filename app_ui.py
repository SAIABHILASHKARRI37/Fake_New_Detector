import streamlit as st
import joblib

# Load the trained model & vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.title("📰 Fake News Detection App")

# User input text box
user_input = st.text_area("Enter news text:", "")

if st.button("Check Authenticity"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Convert input text to feature vector
        transformed_text = vectorizer.transform([user_input])
        
        # Predict using the loaded model
        prediction = model.predict(transformed_text)[0]

        # Display result
        if prediction == 1:
            st.error("🚨 This is Real News!")
        else:
            st.success("✅ This is Fake News!")
