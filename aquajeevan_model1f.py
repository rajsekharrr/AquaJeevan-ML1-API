import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from datetime import timedelta, date
from collections import defaultdict
import joblib # <-- NEW: Import for saving/loading the model
import os

# --- 1. DATA GENERATION FUNCTION ---
def generate_water_quality_data(num_rows=2000):
    """
    Generates a synthetic dataset for water quality monitoring.
    """
    print("Generating a large synthetic dataset for training...")

    # Define ranges for parameters
    param_ranges = {
        'turbidity': {'min': 0.5, 'max': 15.0},
        'pH': {'min': 5.5, 'max': 9.0},
        'temperature': {'min': 15.0, 'max': 35.0},
        'conductivity': {'min': 50.0, 'max': 1500.0},
        'tds': {'min': 100.0, 'max': 1000.0},
        'chloramines': {'min': 0.0, 'max': 5.0},
        'sulfate': {'min': 50.0, 'max': 500.0},
        'rainfall_7d': {'min': 0.0, 'max': 300.0},
        'flow': {'min': 0.0, 'max': 60.0}
    }

    # Generate random data points within specified ranges
    data = {
        'village_id': np.random.choice([f'V{i:03d}' for i in range(1, 21)], size=num_rows),
        'date': [date(2024, 1, 1) + timedelta(days=np.random.randint(0, 365*2)) for _ in range(num_rows)],
        'turbidity': np.random.uniform(param_ranges['turbidity']['min'], param_ranges['turbidity']['max'], num_rows),
        'pH': np.random.uniform(param_ranges['pH']['min'], param_ranges['pH']['max'], num_rows),
        'temperature': np.random.uniform(param_ranges['temperature']['min'], param_ranges['temperature']['max'], num_rows),
        'conductivity': np.random.uniform(param_ranges['conductivity']['min'], param_ranges['conductivity']['max'], num_rows),
        'tds': np.random.uniform(param_ranges['tds']['min'], param_ranges['tds']['max'], num_rows),
        'chloramines': np.random.uniform(param_ranges['chloramines']['min'], param_ranges['chloramines']['max'], num_rows),
        'sulfate': np.random.uniform(param_ranges['sulfate']['min'], param_ranges['sulfate']['max'], num_rows),
        'rainfall_7d': np.random.uniform(param_ranges['rainfall_7d']['min'], param_ranges['rainfall_7d']['max'], num_rows),
        'flow': np.random.uniform(param_ranges['flow']['min'], param_ranges['flow']['max'], num_rows)
    }

    df = pd.DataFrame(data)
    return df

# --- 2. LABELING FUNCTIONS (Rule-based) ---
def label_water_safety(df):
    """
    Labels the water as safe (0) or unsafe (1) based on a set of rules.
    """
    print("Labeling data based on water quality standards...")
    df['water_unsafe'] = 0

    # Rules to mark water as UNSAFE (1)
    df.loc[df['turbidity'] > 5, 'water_unsafe'] = 1
    df.loc[(df['pH'] < 6.5) | (df['pH'] > 8.5), 'water_unsafe'] = 1
    df.loc[df['temperature'] > 30, 'water_unsafe'] = 1
    df.loc[df['conductivity'] > 1200, 'water_unsafe'] = 1
    df.loc[df['tds'] > 500, 'water_unsafe'] = 1
    df.loc[df['chloramines'] > 4, 'water_unsafe'] = 1
    df.loc[df['sulfate'] > 400, 'water_unsafe'] = 1
    df.loc[df['rainfall_7d'] > 100, 'water_unsafe'] = 1
    df.loc[(df['flow'] < 10) | (df['flow'] > 50), 'water_unsafe'] = 1

    return df

def predict_likely_diseases_with_reasons(row):
    """
    Predicts the most likely water-borne diseases and the reasons for their likelihood.
    This is a rule-based function, not a trained ML model.
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
        return [{"disease": "No specific diseases likely", "reasons": ["Water quality parameters are within safe limits."]}]

    # Format the output into a more readable list of dicts
    output = []
    for disease, reasons in disease_reasons.items():
        output.append({
            "disease": disease,
            "reasons": reasons
        })
    return output

# Helper function to get a random float from a range
def get_random_value(param_name):
    param_ranges = {
        'turbidity': {'min': 0.5, 'max': 15.0},
        'pH': {'min': 5.5, 'max': 9.0},
        'temperature': {'min': 15.0, 'max': 35.0},
        'conductivity': {'min': 50.0, 'max': 1500.0},
        'tds': {'min': 100.0, 'max': 1000.0},
        'chloramines': {'min': 0.0, 'max': 5.0},
        'sulfate': {'min': 50.0, 'max': 500.0},
        'rainfall_7d': {'min': 0.0, 'max': 300.0},
        'flow': {'min': 0.0, 'max': 60.0}
    }
    return np.random.uniform(param_ranges[param_name]['min'], param_ranges[param_name]['max'])

# --- MAIN SCRIPT ---
if __name__ == '__main__':
    try:
        # Step 1: Generate the dataset
        df = generate_water_quality_data(num_rows=2000)

        # Step 2: Apply the labeling function
        df = label_water_safety(df)

        # Save the dataset to a CSV file for documentation
        csv_filename = "water_quality_data.csv"
        df.to_csv(csv_filename, index=False)
        print(f"\nDataset saved to '{csv_filename}'.")

        # Step 3: Prepare the data for the ML model
        features = ['turbidity', 'pH', 'temperature', 'conductivity', 'tds', 'chloramines', 'sulfate', 'rainfall_7d', 'flow']
        X = df[features]
        y = df['water_unsafe']

        # Step 4: Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        print("\nTraining a RandomForestClassifier model...")
        # Step 5: Initialize and train the ML model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Step 6: Evaluate the model
        y_pred = model.predict(X_test)

        print("\n--- Model Evaluation ---")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Safe', 'Unsafe']))

        # ===================================================================
        # STEP 7: SAVE THE TRAINED MODEL AND METADATA FOR DEPLOYMENT (NEW)
        # ===================================================================

        model_filename = 'water_safety_model.joblib'
        metadata_filename = 'water_metadata.joblib'

        # 7a. Save the model object
        joblib.dump(model, model_filename)
        print(f"\n✅ Trained Model saved as '{model_filename}'")

        # 7b. Save the features (metadata) required by the model
        model_metadata = {
            'features': features,
            # We don't save output classes here, as it's binary (0/1) but store the rule-based prediction function
            'disease_predictor': predict_likely_diseases_with_reasons
        }
        joblib.dump(model_metadata, metadata_filename)
        print(f"✅ Model Metadata saved as '{metadata_filename}'")
        
        # ===================================================================
        # Interactive Prediction App (Optional after model saved)
        # ===================================================================

        print("\n--- Interactive Prediction App ---")
        print("Enter water quality data or leave an entry blank to use a random value.")

        # ... (Rest of the interactive input loop and prediction display remains the same)
        user_input = {}
        input_prompts = {
            'turbidity': 'Enter Turbidity (NTU): ',
            'pH': 'Enter pH: ',
            'temperature': 'Enter Temperature (°C): ',
            'conductivity': 'Enter Conductivity (µS/cm): ',
            'tds': 'Enter TDS (mg/L): ',
            'chloramines': 'Enter Chloramines (mg/L): ',
            'sulfate': 'Enter Sulfate (mg/L): ',
            'rainfall_7d': 'Enter 7-day Rainfall (mm): ',
            'flow': 'Enter Flow rate (L/min): '
        }

        for param, prompt in input_prompts.items():
            while True:
                user_str = input(prompt).strip()
                if not user_str:
                    user_input[param] = get_random_value(param)
                    print(f"  > No input given. Using random value: {user_input[param]:.2f}")
                    break
                else:
                    try:
                        user_input[param] = float(user_str)
                        break
                    except ValueError:
                        print("Invalid input. Please enter a number or leave it blank.")

        new_data_point = pd.DataFrame([user_input])

        ml_prediction = model.predict(new_data_point)
        # Pass the full row to the rule-based predictor
        disease_predictions = predict_likely_diseases_with_reasons(new_data_point.iloc[0])

        print("\n--- Prediction Results ---")

        if ml_prediction[0] == 1:
            print("Water Status: UNSAFE")
            print("The machine learning model predicts that this water sample is UNSAFE for drinking.")
        else:
            print("Water Status: SAFE")
            print("The machine learning model predicts that this water sample is SAFE for drinking.")

        print("\n--- Likely Disease Outbreaks and Reasons ---")
        for pred in disease_predictions:
            print(f"Disease: {pred['disease']}")
            for reason in pred['reasons']:
                print(f"  - Reason: {reason}")

        print("\n--- Actionable Recommendations for Locality ---")
        if ml_prediction[0] == 1:
            if 'turbidity' in new_data_point.columns and new_data_point.iloc[0]['turbidity'] > 5:
                print("- Recommendation for Turbidity: Advise the community to boil water before consumption to kill pathogens shielded by suspended particles. Investigate the source of runoff and erosion upstream.")
            if 'pH' in new_data_point.columns and (new_data_point.iloc[0]['pH'] < 6.5 or new_data_point.iloc[0]['pH'] > 8.5):
                print("- Recommendation for pH: Check for chemical contamination and adjust the water treatment process to neutralize acidity or alkalinity. Advise residents to use water purifiers.")
            if 'temperature' in new_data_point.columns and new_data_point.iloc[0]['temperature'] > 30:
                print("- Recommendation for Temperature: Investigate the source of heat, which could be from industrial discharge. High temperature promotes bacterial growth.")
            if 'chloramines' in new_data_point.columns and new_data_point.iloc[0]['chloramines'] < 0.5:
                print("- Recommendation for Chloramines: This indicates a failure in the disinfection system. The treatment facility must immediately increase chlorination to a safe residual level.")
            if 'rainfall_7d' in new_data_point.columns and new_data_point.iloc[0]['rainfall_7d'] > 150:
                print("- Recommendation for Rainfall: Issue a public health advisory warning residents about potential contamination from surface runoff. Advise them to boil water and improve drainage systems to prevent contamination.")
            if 'flow' in new_data_point.columns and (new_data_point.iloc[0]['flow'] < 10 or new_data_point.iloc[0]['flow'] > 50):
                print("- Recommendation for Flow: This suggests an infrastructure issue. A low flow indicates a blockage, and a high flow could mean a pipe leak. The utility department should inspect and repair the pipeline network.")
        else:
            print("- All parameters are currently within safe limits. Continue regular monitoring.")

    except Exception as e:
        # If the script fails, print the error and exit gracefully
        print(f"\nAn error occurred: {e}")
        # Optionally, print a traceback for debugging: import traceback; traceback.print_exc()
        print("Exiting application.")
