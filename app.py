from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__) # Creates a Flask application instance.


# Load model & vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/')
def home():
    return "Fake News Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure request contains JSON
        if not request.is_json:
            return jsonify({"error": "Request must be in JSON format"}), 400
        
        data = request.get_json()

        # Validate input
        if "text" not in data:
            return jsonify({"error": "Missing 'text' field in request"}), 400

        text = data["text"]

        # Convert text to features
        transformed_text = vectorizer.transform([text])

        # Predict
        prediction = model.predict(transformed_text)[0]

        # Convert NumPy int64 to Python int
        return jsonify({"prediction": int(prediction)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # Runs on all network interfaces
