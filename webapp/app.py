
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
