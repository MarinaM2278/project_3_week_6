
import flask
from flask import Flask
import pandas as pd

app = flask.Flask(__name__)
#pipe = pickle.load(open('model/pipe.pkl', 'rb'))

@app.route('/page')
def page():
    with open ("page.html", "r") as page:
        return page.read()


@app.route('/result', methods=['POST', 'GET'])
def result():
    '''Gets prediction using the HTML form'''

    if flask.request.method == 'POST':

        inputs = flask.request.form

        name = inputs['name']
        location = inputs['location']
        year = inputs['year']
        kilometers_driven = inputs['kilometers_driven']
        fuel_type = inputs['fuel_type']
        transmission = inputs['transmission']
        owner_type = inputs['owner_type']
        seats = inputs['seats']



        data = pd.DataFrame([{
            'name' : name,
            'location' : location,
            'year' : year,
            'kilometers_driven' : kilometers_driven,
            'fuel_type' : fuel_type,
            'transmission' : transmission,
            'owner_type' : owner_type,
            'seats' : seats}])


        pred = pipe.predict(data)[0]
        results = {'price': round(pred, 6)}
        return flask.jsonify(results)
