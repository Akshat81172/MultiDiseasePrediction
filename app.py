from flask import Flask, render_template, request
import numpy as np
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load both saved models
diabetes_model = pickle.load(open("diabetes_model.sav", "rb"))
heart_disease_model = pickle.load(open("heart_disease_model.sav", "rb"))


# Home route with links to both models
@app.route('/')
def index():
    return render_template('index.html')


# Diabetes Prediction Route
@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'POST':
        # Get form values for Diabetes prediction
        Pregnancies = float(request.form['Pregnancies'])
        Glucose = float(request.form['Glucose'])
        SkinThickness = float(request.form['SkinThickness'])
        Insulin = float(request.form['Insulin'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])

        # Convert inputs to numpy array
        input_data = np.array([[Pregnancies, Glucose, SkinThickness, Insulin, DiabetesPedigreeFunction]])

        # Predict using the diabetes model
        prediction = diabetes_model.predict(input_data)

        # Interpret prediction
        if prediction[0] == 0:
            result = "The person is non-diabetic."
        else:
            result = "The person is diabetic."

        return render_template('diabetes.html', prediction_text=result)

    return render_template('diabetes.html')


# Heart Disease Prediction Route
@app.route('/heart_disease', methods=['GET', 'POST'])
def heart_disease():
    if request.method == 'POST':
        # Get form values for Heart Disease prediction
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        # Convert inputs to numpy array
        input_data = np.array([[cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        # Predict using the heart disease model
        prediction = heart_disease_model.predict(input_data)

        # Interpret prediction
        if prediction[0] == 0:
            result = "This person's heart is healthy."
        else:
            result = "This person's heart is at risk of disease."

        return render_template('heart_disease.html', prediction_text=result)

    return render_template('heart_disease.html')


if __name__ == '__main__':
    app.run(debug=True)



