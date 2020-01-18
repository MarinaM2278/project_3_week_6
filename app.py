import pickle

from flask import Flask, render_template, request
import pandas as pd


app = Flask(__name__, template_folder='templates')
pipe = pickle.load(open('model/pipe.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    args = request.form
    data = DataFrameMapper([
    (['name'], [LabelBinarizer()]),
    (['location'], [LabelBinarizer()]),
    (['year'], [StandardScaler()]),
    (['kilometers_driven'], [SimpleImputer(), StandardScaler()]),
    (['fuel_type'], [CategoricalImputer(), LabelBinarizer()]),
   (['transmission'],[CategoricalImputer(), LabelBinarizer()]),
    (['owner_type'], [SimpleImputer(), StandardScaler()]),
    (['seats'], [SimpleImputer(), StandardScaler()]),
    ], df_out= True)

    id = int(pipe.predict(data))

    return render_template(
        'result.html',
        price=df.loc[id]['price']

if __name__ == '__main__':
    app.run(port=5000, debug=True)
