from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np


app = Flask(__name__)

# Load model, scaler, and encoder
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Home route to serve HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [
        data['acidity'], data['caffeine'], data['aroma'], data['texture'],
        data['bitterness'], data['sweetness'], data['body'], data['aftertaste'],
        data['color']
    ]
    roast_level = data['roast_level']

    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    roast_level_encoded = label_encoder.transform([roast_level])
    final_features = np.hstack([features_scaled, roast_level_encoded.reshape(-1, 1)])
    prediction = model.predict(final_features)[0]

    return jsonify({'quality_label': prediction})

if __name__ == '__main__':
    app.run(debug=True)
