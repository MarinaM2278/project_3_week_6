import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn import preprocessing

from sklearn_pandas import DataFrameMapper, CategoricalImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.preprocessing import LabelBinarizer, StandardScaler, LabelEncoder

import sklearn.decomposition, sklearn.linear_model, sklearn.pipeline, sklearn.metrics
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import pickle
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error

from sklearn.pipeline import Pipeline

df = pd.read_csv('data/train-data.csv')

target = 'Price'
y = df[target]
X = df.drop(target, axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=42)


X_train = X_train.drop('New_Price', axis = 1)
X_test = X_test.drop('New_Price', axis = 1)

X_train['Engine']= X_train['Engine'].astype('str')
X_test['Engine']= X_test['Engine'].astype('str')

X_train['Engine'] = X_train['Engine'].map(lambda x: x.lstrip('-CC').rstrip('aAbBcC'))
X_test['Engine'] = X_test['Engine'].map(lambda x: x.lstrip('-CC').rstrip('aAbBcC'))

split_train = X_train['Power'].str.split(' ', 1, expand= True)
split_test = X_test['Power'].str.split(' ', 1, expand= True)

X_train_split = X_train.assign(first_part=split_train[0], last_part=split_train[1])
X_test_split = X_test.assign(first_part=split_test[0], last_part=split_test[1])

X_train_split.drop('Power', 1, inplace=True)
X_test_split.drop('Power', 1, inplace=True)

X_train_split.drop('last_part', 1, inplace=True)
X_test_split.drop('last_part', 1, inplace=True)

X_train = X_train_split
X_test = X_test_split

X_train = X_train.rename(columns = {'first_part':'power'})
X_test = X_test.rename(columns = {'first_part':'power'})

X_train['power'] = X_train['power'].astype('str')
X_test['power'] = X_test['power'].astype('str') ###


X_train['Mileage'] = X_train['Mileage'].astype('str')
X_test['Mileage'] = X_test['Mileage'].astype('str')

split1_train = X_train['Mileage'].str.split(' ', 1, expand= True)
split1_test = X_test['Mileage'].str.split(' ', 1, expand= True)
X_train_split2 = X_train.assign(first_part=split1_train[0], last_part=split1_train[1])
X_test_split2 = X_test.assign(first_part=split1_test[0], last_part=split1_test[1])
X_train_split2.drop('Mileage', 1, inplace=True)
X_test_split2.drop('Mileage', 1, inplace=True)

X_train_split2.drop('last_part', 1, inplace=True)
X_test_split2.drop('last_part', 1, inplace=True)

X_train = X_train_split2
X_test = X_test_split2

X_train = X_train.rename(columns = {'first_part':'mileage'})
X_test = X_test.rename(columns = {'first_part':'mileage'})

split1_train = X_train['Name'].str.split(' ', 1, expand= True)
split1_test = X_test['Name'].str.split(' ', 1, expand= True)

X_train_split2 = X_train.assign(first_part=split1_train[0], last_part=split1_train[1])
X_test_split2 = X_test.assign(first_part=split1_test[0], last_part=split1_test[1])

X_train_split2.drop('Name', 1, inplace=True)
X_test_split2.drop('Name', 1, inplace=True)

X_train_split2.drop('last_part', 1, inplace=True)
X_test_split2.drop('last_part', 1, inplace=True)

X_train = X_train_split2
X_test = X_test_split2

X_train = X_train.rename(columns = {'first_part':'name'})
X_test = X_test.rename(columns = {'first_part':'name'})


X_train['Owner_Type'] = X_train['Owner_Type'].apply({'First':1, 'Second':2, 'Third' : 3, 'Fourth & Above': 4}.get)
X_test['Owner_Type'] = X_test['Owner_Type'].apply({'First':1, 'Second':2, 'Third' : 3, 'Fourth & Above': 4}.get)

X_train['Transmission'] = X_train['Transmission'].apply({'Manual': 0, 'Automatic': 1}.get)
X_test['Transmission'] = X_test['Transmission'].apply({'Manual': 0, 'Automatic': 1}.get)


X_test.columns = map(str.lower, X_test.columns)

X_train.columns = map(str.lower, X_train.columns)

X_train.drop('power',  1, inplace=True)
X_test.drop('power',  1, inplace=True)

X_train.drop('engine',  1, inplace=True)
X_test.drop('engine',  1, inplace=True)

X_train.drop('mileage',  1, inplace=True)
X_test.drop('mileage',  1, inplace=True)

y_train = y_train * 0.014      ###conversion to US $

X_train.rename({"unnamed: 0":"a"}, axis="columns", inplace=True)
X_train.drop(["a"], axis=1, inplace=True)

X_test.rename({"unnamed: 0":"a"}, axis="columns", inplace=True)
X_test.drop(["a"], axis=1, inplace=True)

le = preprocessing.LabelEncoder()

le.fit( X_train['name'])
le.transform(X_train['name'])

le.fit( X_train['location'])
le.transform(X_train['location'])


mapper = DataFrameMapper([
    (['name'], [LabelBinarizer()]),
    (['location'], [LabelBinarizer()]),
    (['year'], [StandardScaler()]),
    (['kilometers_driven'], [SimpleImputer(), StandardScaler()]),
    (['fuel_type'], [CategoricalImputer(), LabelBinarizer()]),
   (['transmission'],[CategoricalImputer(), LabelBinarizer()]),
    (['owner_type'], [SimpleImputer(), StandardScaler()]),
    (['seats'], [SimpleImputer(), StandardScaler()]),
    ], df_out= True)

Z_train= mapper.fit_transform(X_train)
Z_test = mapper.transform(X_test)


model = LinearRegression()
model.fit(Z_train, y_train)
model.score(Z_train, y_train)
model.score(Z_test, y_test)

i = 1
X_test.iloc[i : i + 1].to_dict(orient="list")
y_test.values[i]

new_data = pd.DataFrame ({
    'name':  ['BMW'],
    'location':  ['Bangalore'],
    'year' : [2000],
    'kilometers_driven' : [120000],
    'fuel_type' : ['Diesel'],
   'transmission': [1],
    'owner_type' : [2],
    'seats' : [5]
})

mapper.transform(new_data)
model.predict(mapper.transform(new_data))[0]

round(model.predict(mapper.transform(new_data))[0], 5)

pipe = Pipeline([("mapper", mapper), ("model", model)])

pipe.predict(new_data)

pickle.dump(pipe, open('pipe.pkl', 'wb'))
del pipe
pipe = pickle.load(open('pipe.pkl', 'rb'))
pipe.predict(new_data)
