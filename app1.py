from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open('medicine_prediction.pkl', 'rb'))

# Home route to render the HTML form
@app.route('/')
def home():
    return render_template('p.html')

# Prediction route to handle form submissions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        data = request.get_json()
        features = [
            float(data['temperature']),
            1 if data['feverSeverity'] == 'mildfever' else 2 if data['feverSeverity'] == 'highfever' else 0,
            float(data['age']),
            1 if data['gender'] == 'male' else 0,
            float(data['bmi']),
            1 if data['headache'] == 'yes' else 0,
            1 if data['bodyAche'] == 'yes' else 0,
            1 if data['fatigue'] == 'yes' else 0,
            1 if data['chronicConditions'] == 'yes' else 0,
            1 if data['allergies'] == 'yes' else 0,
            1 if data['smoking'] == 'yes' else 0,
            1 if data['alcohol'] == 'yes' else 0,
            float(data['humidity']),
            float(data['aqi']),
            2 if data['physicalActivity'] == 'active' else 1 if data['physicalActivity'] == 'moderate' else 0,
            2 if data['diet'] == 'non-vegetarian' else 1 if data['diet'] == 'vegetarian' else 0,
            float(data['heartRate']),
            2 if data['bp'] == 'high' else 1 if data['bp'] == 'low' else 0,
        ]

        # Predict using the loaded model
        prediction = model.predict([features])[0]

        # Map the prediction to a medicine recommendation
        medicine = "Take Paracetamol" if prediction == 1 else "Take Ibuprofen"

        # Return result
        return jsonify({'medicine': medicine})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
