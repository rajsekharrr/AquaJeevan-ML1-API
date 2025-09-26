from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

# 🔑 FIX: Import the function directly from the module where it is defined.
# The metadata file (water_metadata.joblib) no longer contains this function.
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

    print("✅ Water Safety Model and metadata loaded successfully.")
    print(f"Expected features: {FEATURES}")

except FileNotFoundError:
    print("FATAL ERROR: water_safety_model.joblib or water_metadata.joblib not found. Ensure they are in the deployment root.")
    # Exiting is appropriate for deployment failure
    exit()
except Exception as e:
    # This catch handles the joblib serialization error if the metadata file was NOT regenerated correctly.
    print(f"FATAL ERROR loading model components: {str(e)}. The most common cause is a function reference saved in a .joblib file. Please re-run the training script.")
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
                # Use the provided data value
                input_data[feat] = data[feat]
            else:
                # 💡 Fallback: Use a random value for features missing from the request payload
                input_data[feat] = np.random.uniform(0.5, 1.0) 
        
        # Create a DataFrame containing all required features
        input_df = pd.DataFrame([input_data])


        # ML prediction
        # Ensure only the expected FEATURES columns are passed to the model
        ml_pred = MODEL.predict(input_df[FEATURES])[0]  # 0 = safe, 1 = unsafe
        ml_pred_label = 'UNSAFE' if ml_pred == 1 else 'SAFE'
        
        # Calculate probability for richer output
        ml_prob = MODEL.predict_proba(input_df[FEATURES])[0][1] # Probability of "unsafe" class (index 1)


        # Rule-based disease prediction
        # Call the imported function directly, passing the single input row (Series)
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
        # Log the error for debugging
        print(f"Prediction Error: {str(e)}") 
        return jsonify({'error': f'An unexpected error occurred during prediction: {str(e)}'}), 500

@app.route('/')
def index():
    return "Water Safety Prediction API is running. Use the /predict_water endpoint (POST) for predictions."

# ----------------- Run Server -----------------
if __name__ == '__main__':
    # Running locally for testing (Gunicorn handles this in production)
    app.run(host='0.0.0.0', port=5000)
