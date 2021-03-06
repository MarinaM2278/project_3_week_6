
import flask
import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__, template_folder='templates')
pipe = pickle.load(open("pipe.pkl", 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    args = request.form
    data = pd.DataFrame({
        'name': [args.get('name')],
        'location': [args.get('location')],
        'year':  [int(args.get('year'))],
        'kilometers_driven' : [int(args.get('kilometers_driven'))],
        'fuel_type': [args.get('fuel_type')],
        'transmission' : [args.get('transmission')],
        'owner_type' : [int(args.get('owner_type'))],
        'seats': [int(args.get('owner_type'))]
        })

    price = np.round(pipe.predict(data)* 100000).astype('int')

    #price = (np.round(price, 1)).astype('int')

    return render_template('result.html', price = price )

if __name__ == '__main__':
    app.run(port=5000, debug=True)
