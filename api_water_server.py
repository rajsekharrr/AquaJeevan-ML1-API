# api_water_server.py
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from collections import defaultdict
import numpy as np

# --- NOTE: The rule-based functions MUST be included here for joblib to load them ---

def predict_likely_diseases_with_reasons(row):
    """
    Predicts the most likely water-borne diseases and the reasons for their likelihood.
    This is a rule-based function, NOT a trained ML model prediction.
    It takes a single row (Series) of water quality data.
    """
    disease_reasons = defaultdict(list)

    # High turbidity often indicates the presence of suspended solids and pathogens
    if row['turbidity'] > 5:
        disease_reasons["cholera"].append("High turbidity can shield pathogens like Vibrio cholerae from disinfection.")
        disease_reasons["typhoid"].append("High turbidity can indicate fecal contamination, a common source of typhoid.")
        disease_reasons["diarrhea"].append("High turbidity is a direct indicator of potential microbial contamination that causes diarrhea.")
        disease_reasons["giardiasis"].append("Giardia cysts can attach to suspended particles in turbid water, interfering with disinfection.")

    # High sulfate can cause digestive issues, including diarrhea
    if row['sulfate'] > 400:
        disease_reasons["diarrhea"].append("High sulfate concentration acts as a laxative, which can lead to diarrhea.")

    # Low chloramines suggest a failure in the disinfection process
    if row['chloramines'] < 0.5:
        disease_reasons["cholera"].append("Insufficient chlorination fails to kill cholera-causing bacteria.")
        disease_reasons["typhoid"].append("Insufficient chlorination allows the spread of typhoid-causing bacteria.")
        disease_reasons["diarrhea"].append("Lack of a disinfectant residual allows a variety of microbes to proliferate.")
        disease_reasons["hepatitis A"].append("Hepatitis A virus can spread through contaminated water if not properly disinfected.")

    # High rainfall leads to surface runoff and contamination of water sources
    if row['rainfall_7d'] > 150:
        disease_reasons["cholera"].append("Heavy rainfall can cause overflow of sewage systems and transport pathogens into water sources.")
        disease_reasons["typhoid"].append("Heavy rainfall increases the risk of fecal contamination from runoff.")
        disease_reasons["diarrhea"].append("High runoff can introduce various fecal pathogens that cause diarrhea.")
        disease_reasons["giardiasis"].append("Heavy rains can wash Giardia cysts from animal waste into drinking water sources.")
        disease_reasons["hepatitis A"].append("Contaminated runoff from heavy rains can be a source of the hepatitis A virus.")

    # Note: Malaria is mosquito-borne, but stagnant water from heavy rain can increase risk
    if row['rainfall_7d'] > 200:
        disease_reasons["malaria"].append("While not water-borne, heavy rainfall and flooding can create stagnant pools of water, which are ideal breeding grounds for malaria-carrying mosquitoes.")

    # Low or high flow indicates infrastructure problems which can lead to contamination
    if row['flow'] < 10:
        disease_reasons["diarrhea"].append("Low flow can indicate a blockage, leading to stagnant water where bacteria can multiply.")
        disease_reasons["cholera"].append("Stagnant water due to low flow can become a breeding ground for bacteria.")

    if row['flow'] > 50:
        disease_reasons["diarrhea"].append("Abnormally high flow can indicate a leak or broken pipe, allowing external contaminants to enter the water system.")
        disease_reasons["typhoid"].append("High-pressure leaks can suck in contaminated groundwater or soil, leading to typhoid outbreaks.")

    if not disease_reasons:
        return [{"disease": "None identified", "reasons": ["Water quality parameters are within safe limits."]}]

    # Format the output into a more readable list of dicts
    output = []
    for disease, reasons in disease_reasons.items():
        output.append({
            "disease": disease,
            "reasons": reasons
        })
    return output


# --- Helper function needed for joblib compatibility (even if not used directly by API) ---
# This ensures that all components referenced during the model saving process are available globally.
def get_random_value(param_name):
     # This function is usually not needed in the API server itself, 
     # but including it resolves potential dependency issues from the joblib load process.
     return 0.0 
     
# ----------------- Initialize Flask -----------------
app = Flask(__name__)

# ----------------- Load Model & Metadata -----------------
# We load the model safely by ensuring all required functions are defined first.
try:
    # Use the global context to ensure the function is recognized during joblib loading
    MODEL = joblib.load('water_safety_model.joblib')
    METADATA = joblib.load('water_metadata.joblib')

    FEATURES = METADATA['features']
    # The function is already globally defined, resolving the error
    DISEASE_FUNC = METADATA['disease_predictor'] 

    print("✅ Water Safety Model and metadata loaded successfully.")
    print(f"Expected features: {FEATURES}")

except FileNotFoundError:
    # Exit gracefully if model files are missing during deployment setup
    print("Error: water_safety_model.joblib or water_metadata.joblib not found.")
    # On Render, this will cause the service to fail startup as expected
    exit()
except Exception as e:
    # Catching the specific joblib reference error if it occurs
    print(f"Error loading model files: {str(e)}")
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

        # Check missing features
        missing = [f for f in FEATURES if f not in data]
        if missing:
            return jsonify({
                'error': 'Missing required features',
                'missing': missing,
                'required': FEATURES
            }), 400

        # Create DataFrame from input data
        input_df = pd.DataFrame([[data[f] for f in FEATURES]], columns=FEATURES)

        # ML prediction
        ml_pred = MODEL.predict(input_df)[0]  # 0 = safe, 1 = unsafe
        # Probability of being the "unsafe" class (index 1)
        ml_prob = MODEL.predict_proba(input_df)[0][1] 

        # Rule-based disease prediction (Uses the function loaded from metadata)
        disease_preds = DISEASE_FUNC(input_df.iloc[0])

        # Prepare final output JSON
        response = {
            'status': 'success',
            'water_status': 'UNSAFE' if ml_pred == 1 else 'SAFE',
            'unsafe_probability': round(float(ml_prob), 3),
            'disease_predictions': disease_preds
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

# Optional health check
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'Water Model API is running'}), 200

# ----------------- Run Server -----------------
# We use port 5001 locally but Render/Gunicorn will handle the port in production
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
