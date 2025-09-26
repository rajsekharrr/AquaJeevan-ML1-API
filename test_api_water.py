# test_api_water.py
import requests
import json

# ----------------- API URL -----------------
# Change this to your deployed URL if not running locally
API_URL = "http://127.0.0.1:5001/predict_water"

# ----------------- Sample Water Data -----------------
sample_data = {
    "turbidity": 6.2,
    "pH": 7.2,
    "temperature": 28.5,
    "conductivity": 800,
    "tds": 400,
    "chloramines": 0.3,
    "sulfate": 420,
    "rainfall_7d": 160,
    "flow": 8
}

# ----------------- HTTP Headers -----------------
headers = {'Content-Type': 'application/json'}

print(f"Sending water data to {API_URL}...")
print("Input:", sample_data)

try:
    # Send POST request
    response = requests.post(API_URL, data=json.dumps(sample_data), headers=headers)

    if response.status_code == 200:
        result = response.json()
        print("\n--- API Prediction Success ---")
        print(f"Water Status: {result['water_status']}")
        print(f"Unsafe Probability: {result['unsafe_probability']:.2f}")

        print("\nLikely Disease Outbreaks and Reasons:")
        for disease_info in result['disease_predictions']:
            print(f"- Disease: {disease_info['disease']}")
            for reason in disease_info['reasons']:
                print(f"    * Reason: {reason}")

    else:
        print("\n--- API Error ---")
        print(f"Status Code: {response.status_code}")
        print(f"Error Response: {response.text}")

except requests.exceptions.ConnectionError:
    print("\nError: Could not connect to the API server.")
    print("Please make sure api_water_server.py is running in a separate terminal.")
