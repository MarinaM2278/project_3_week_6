
import flask
import pickle
from flask import Flask, request, jsonify, render_template
import pandas as pd

app = flask.Flask(__name__)


pipe = pickle.load(open("pipe.pkl", 'rb'))

@app.route('/')
def home():
    return render_template("index_html")

@app.route('/predict', methods=['POST'])
def predict():
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
})


    list = [x for x in request.form.values()]
    final_features = [np.array(list)]
    prediction = pipe.predict(final_features)
    output = int(round(pipe.predict(data)[0], 5))
    return render_template ('index_html', prediction_text = 'Yor Car price will be $: {}', format(output)')

@app.route('/predict_api', methods=['POST'])
    def predict_api():
        data = request.get_json(force = True)
        prediction = pipe.predict([np.arraylist(data.values))])
        output = prediction[0]
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
