import logging
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shap

# Initialize Flask application
app = Flask(__name__)

# Setup logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Dummy model and scaler (replace with your actual trained model and scaler)
model = RandomForestClassifier(n_estimators=100, random_state=42)
scaler = StandardScaler()

# Simulating training of the model
data = pd.DataFrame({
    'age': np.random.randint(20, 80, size=100),
    'blood_pressure': np.random.randint(90, 180, size=100),
    'heart_rate': np.random.randint(60, 120, size=100),
    'glucose_level': np.random.randint(70, 200, size=100),
    'cholesterol': np.random.randint(150, 250, size=100),
    'weight': np.random.randint(45, 100, size=100),
    'bmi': np.random.uniform(18.5, 40, size=100),
    'family_history': np.random.randint(0, 2, size=100)
})

# Target variable (Disease risk)
data['disease_risk'] = np.random.randint(0, 2, size=100)

# Features and target
X = data.drop('disease_risk', axis=1)
y = data['disease_risk']

# Train the model (for demonstration purposes)
X_scaled = scaler.fit_transform(X)
model.fit(X_scaled, y)

@app.route('/')
def index():
    return "Flask API is running"

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Log the incoming request for debugging
        logging.debug("Received request data: %s", request.get_json())

        # Get data from the request
        data = request.get_json()

        # Ensure all required fields are present
        required_fields = ['age', 'blood_pressure', 'heart_rate', 'glucose_level', 'cholesterol', 'bmi', 'family_history']
        for field in required_fields:
            if field not in data:
                logging.error("Missing required field: %s", field)
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Prepare features for prediction
        features = np.array([
            data['age'],
            data['blood_pressure'],
            data['heart_rate'],
            data['glucose_level'],
            data['cholesterol'],
            data['bmi'],
            data['family_history']
        ]).reshape(1, -1)

        logging.debug("Prepared features for prediction: %s", features)

        # Scale the features
        features_scaled = scaler.transform(features)
        logging.debug("Scaled features: %s", features_scaled)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        logging.debug("Prediction result: %s", prediction)

        # SHAP values for model interpretability
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features_scaled)

        # Select SHAP values for the positive class (Disease risk)
        shap_explanation = shap_values[1].tolist()  # For binary classification (class 1)

        # Log SHAP explanation
        logging.debug("SHAP explanation: %s", shap_explanation)

        # Return the prediction and SHAP explanation
        return jsonify({
            'prediction': prediction,
            'shapExplanation': shap_explanation
        })

    except Exception as e:
        # Log any exception that occurs
        logging.error("Error in prediction: %s", str(e))
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
