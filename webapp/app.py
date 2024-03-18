'''from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load or define the trained model
model = None
# Load the trained model if it exists
try:
    model = joblib.load('model/logisticmodel.pkl')
except FileNotFoundError:
    print("Model file not found. Make sure the model is trained and saved before running the Flask application.")

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if model is None:
            return "Model not loaded. Please train and save the model before making predictions."
        
        Review_text = request.form['Review_text']
        prediction = model.predict([Review_text])
        sentiment = 'Positive' if prediction == 1 else 'Negative'
        return render_template('result.html', Review=Review_text, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
'''
'''
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load('model/naive_bayes_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    prediction = model.predict([text])
    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")'''
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load('model/naive_bayes_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    prediction = model.predict([text])
    prediction_proba = model.predict_proba([text])[0]
    positive_percentage = round(prediction_proba[1] * 100, 2)
    negative_percentage = round(prediction_proba[0] * 100, 2)
    return render_template('index.html', prediction=prediction[0], positive_percentage=positive_percentage, negative_percentage=negative_percentage)

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")
