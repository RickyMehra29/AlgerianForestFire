import pickle
from flask import Flask, request, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
scalar = pickle.load(open("Algerian_scaler.pkl", "rb"))
Algerian_log_model = pickle.load(open("Algerian_log_model.pkl", "rb"))
Algerian_liblinear_model = pickle.load(open("Algerian_liblinear_model.pkl", "rb"))
Algerian_rf_model  = pickle.load(open("Algerian_rf_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1,-1))
    output = Algerian_log_model.predict(final_input)[0]
    print(output)
    if data[-1] == 0.0:
        selected_region = 'Bejaia'
    else:
        selected_region = 'Sidi-Bel Abbes'
    if output == 0:
        result = 'Fire'
    else:
        result = 'No Fire'
    return render_template('home.html', output_text="Algeria Region - {} will have {}.".format(selected_region, result))


if __name__ == '__main__':
    app.run(debug=True)
