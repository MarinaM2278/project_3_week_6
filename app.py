
import flask
import pickle
from flask import Flask
import pandas as pd

app = flask.Flask(__name__)


#-------- MODEL GOES HERE -----------#


pipe = pickle.load(open("pipe.pkl", 'rb'))

#-------- ROUTES GO HERE -----------#

@app.route('/result', methods=['GET'])
def predict_car_price(name):
    args = dict(flask.request.args)
    data = pd.DataFrame({
        'name': [name],
        'location': [location],
        'year':  [int(args.get('year'))],
        'kilometers_driven' : [int(args.get('kilometers_driven'))],
        'fuel_type': [fuel_type],
        'transmission' : [int(args.get('transmission'))],
        'owner_type' : [int(args.get('owner_type'))],
        'seats': [int(args.get('owner_type'))]


    q = int(round(pipe.predict(data)[0], 5))
    prediction = {'price': q}
    return flask.jsonify(prediction)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
