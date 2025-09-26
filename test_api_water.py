from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

# 🔑 CRITICAL: Import the rule-based function directly from the module where it is defined.
# This avoids the Gunicorn serialization error.
from aquajeevan_model1f import predict_likely_diseases_with_reasons 

# ----------------- Initialize Flask -----------------
app = Flask(__name__)

# ----------------- Load Model & Metadata -----------------
try:
    # 1. Load the ML model (your classifier)
    MODEL = joblib.load('water_safety_model.joblib')
    
    # 2. Load the metadata (which now ONLY contains the 'features' list)
    METADATA = joblib.load('water_metadata.joblib')

    # Get features list from the clean metadata
    FEATURES = METADATA['features']
    # NOTE: There is NO line attempting to load 'disease_predictor' from METADATA.

    print("✅ Water Safety Model and metadata loaded successfully.")
    print(f"Expected features: {FEATURES}")

except FileNotFoundError:
    print("FATAL ERROR: water_safety_model.joblib or water_metadata.joblib not found. Ensure they are in the deployment root.")
    exit()
except Exception as e:
    # This catch confirms the model is loaded correctly.
    print(f"FATAL ERROR loading model components: {str(e)}. The most common cause is a function reference saved in a .joblib file.")
    exit()

# ----------------- API Endpoints -----------------

@app.route('/predict_water', methods=['POST'])
def predict_water():
    """
    POST endpoint to predict water safety and likely disease outbreaks.
    Expects JSON with water parameters.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Create DataFrame for ML prediction, filling missing features with a fallback value
        input_data = {}
        for feat in FEATURES:
            if feat in data:
                input_data[feat] = data[feat]
            else:
                # Use a random fallback value for features missing from the request payload
                input_data[feat] = np.random.uniform(0.5, 1.0) 
        
        input_df = pd.DataFrame([input_data])


        # ML prediction
        # Ensure only the expected FEATURES columns are passed to the model
        ml_pred = MODEL.predict(input_df[FEATURES])[0]  # 0 = safe, 1 = unsafe
        ml_pred_label = 'UNSAFE' if ml_pred == 1 else 'SAFE'
        
        # Calculate probability
        ml_prob = MODEL.predict_proba(input_df[FEATURES])[0][1] 

        # Rule-based disease prediction: call the directly imported function
        disease_predictions = predict_likely_diseases_with_reasons(input_df.iloc[0])


        # Prepare final output JSON
        response = {
            'status': 'success',
            'water_status': ml_pred_label,
            'unsafe_probability': round(float(ml_prob), 3),
            'disease_predictions': disease_predictions
        }

        return jsonify(response), 200

    except Exception as e:
        print(f"Prediction Error: {str(e)}") 
        return jsonify({'error': f'An unexpected error occurred during prediction: {str(e)}'}), 500

@app.route('/')
def index():
    return "Water Safety Prediction API is running. Use the /predict_water endpoint (POST) for predictions."

# ----------------- Run Server -----------------
if __name__ == '__main__':
    # Local port set to 5001 to match your test_api_water.py script
    app.run(host='0.0.0.0', port=5001)
