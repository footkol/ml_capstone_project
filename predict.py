import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from math import exp
import numpy as np
from flask import Flask, request, jsonify

with open('lightgbm_model.bin', 'rb') as f_in:
    model = pickle.load(f_in)


with open('scaler_model.bin', 'rb') as f_in:
    scaler, lmbda = pickle.load(f_in)

app = Flask('predict')

@app.route('/predict', methods=['POST'])


def predict():
    location = request.get_json()

    location_series = pd.Series(location)
    location_scaled = scaler.fit_transform(location_series.values.reshape(1,-1))

    y_pred = model.predict(location_scaled)
    result = {"traffic_volume": float(y_pred)}

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)


#y_inversed = scaler.inverse_transform(y_pred.reshape(1, -1))

#y_inverse = invert_yeojhonson(y_inversed, lmbda)  
#print(y_inverse) 



