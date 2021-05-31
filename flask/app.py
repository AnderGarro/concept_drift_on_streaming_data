from flask import Flask
from flask import jsonify, request
import time
import uuid
import numpy as np
import pickle
import json
import pandas as pd
import gc

from urllib.parse import urlparse
from functions_sc import *

app = Flask(__name__)
       
timestamp = 0
y_pred = get_curve(timestamp,noise_amp=50)
print(y_pred)

@app.route('/get_curve_zero', methods=['GET', 'POST'])
def get_curve_zero():
    timestamp = 0
    return jsonify(y_pred.values[0].tolist())

@app.route('/get_curve_time', methods=['GET', 'POST'])
def get_curve_time():
    d = request.get_json()
    print(d)
    timestamp = int(d)
    y_pred = get_curve(timestamp,noise_amp=50)
    print(y_pred)
    return jsonify(y_pred.values[0].tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)



