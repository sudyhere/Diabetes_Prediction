import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

regmodel = pickle.load(open('classification_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('predict.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    def encoder_smoking(label):
        if label == 'Never':
            return 0
        elif label == 'Former':
            return 1
        else:
            return 2

    def encode_gender(label):
        if label == 'Female':
            return 0
        elif label == 'Male':
            return 1
        else:
            return 2

    def encode_heart_disease(label):
        if label == 'No':
            return 0
        else:
            return 1

    def encode_hypertension(label):
        if label == 'No':
            return 0
        else:
            return 1

    data = request.json['data']
    print(data)

    # Extract values from data dictionary
    gender = data['gender']
    age = data['age']
    hypertension = data['hypertension']
    heart_disease = data['heart_disease']
    smoking_history = data['smoking_history']
    bmi = data['bmi']
    HbA1c_level = data['HbA1c_level']
    blood_glucose_level = data['blood_glucose_level']

    # Apply label encoding
    gender = encode_gender(gender)
    hypertension = encode_hypertension(hypertension)
    heart_disease = encode_heart_disease(heart_disease)
    smoking_history = encoder_smoking(smoking_history)

    # Create input array
    input_data = np.array([gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level])

    input_data = input_data.reshape(1, -1)  # Reshape the data

    output = regmodel.predict(input_data)
    print(output)
    output = output.tolist()  # Convert ndarray to list
    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True)
