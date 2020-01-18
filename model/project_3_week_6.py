#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

import catboost as cb
from catboost import Pool, CatBoostRegressor
from sklearn.dummy import DummyRegressor

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

from sklearn_pandas import DataFrameMapper, CategoricalImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.preprocessing import LabelBinarizer, StandardScaler, LabelEncoder

import sklearn.decomposition, sklearn.linear_model, sklearn.pipeline, sklearn.metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import pickle
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error

pd.set_option('display.max_rows', 100)


# In[ ]:


from sklearn.pipeline import Pipeline, make_pipeline


# In[ ]:


df = pd.read_csv('data/train-data.csv')


# In[ ]:


df.describe().T


# In[ ]:


df.isnull().sum()


# In[ ]:


target = 'Price'
y = df[target]
X = df.drop(target, axis = 1)


# In[ ]:





# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=42)


# In[ ]:


X_train.head()


# In[ ]:


X_train = X_train.drop('New_Price', axis = 1)
X_test = X_test.drop('New_Price', axis = 1)


# In[ ]:


X_train['Engine']= X_train['Engine'].astype('str')
X_test['Engine']= X_test['Engine'].astype('str')


# In[ ]:


X_train['Engine'] = X_train['Engine'].map(lambda x: x.lstrip('-CC').rstrip('aAbBcC'))
X_test['Engine'] = X_test['Engine'].map(lambda x: x.lstrip('-CC').rstrip('aAbBcC'))


# In[ ]:


#df['Engine']= df['Engine'].astype(float)


# In[ ]:


split_train = X_train['Power'].str.split(' ', 1, expand= True)
split_test = X_test['Power'].str.split(' ', 1, expand= True)


# In[ ]:


X_train_split = X_train.assign(first_part=split_train[0], last_part=split_train[1])
X_test_split = X_test.assign(first_part=split_test[0], last_part=split_test[1])


# In[ ]:


X_train_split


# In[ ]:


X_train_split.drop('Power', 1, inplace=True)
X_test_split.drop('Power', 1, inplace=True)

X_train_split.drop('last_part', 1, inplace=True)
X_test_split.drop('last_part', 1, inplace=True)

X_train = X_train_split
X_test = X_test_split


# In[ ]:


X_train = X_train.rename(columns = {'first_part':'power'})
X_test = X_test.rename(columns = {'first_part':'power'})


# In[ ]:


X_train['power'] = X_train['power'].astype('str') 
X_test['power'] = X_test['power'].astype('str') ###


# In[ ]:


X_train['Mileage'] = X_train['Mileage'].astype('str')
X_test['Mileage'] = X_test['Mileage'].astype('str')


# In[ ]:


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


# In[ ]:


X_train = X_train.rename(columns = {'first_part':'mileage'})
X_test = X_test.rename(columns = {'first_part':'mileage'})


# In[ ]:


X_train


# In[ ]:


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


# In[ ]:


X_train['Owner_Type'].value_counts()


# In[ ]:


X_train['Owner_Type'] = X_train['Owner_Type'].apply({'First':1, 'Second':2, 'Third' : 3, 'Fourth & Above': 4}.get)
X_test['Owner_Type'] = X_test['Owner_Type'].apply({'First':1, 'Second':2, 'Third' : 3, 'Fourth & Above': 4}.get)


# In[ ]:


X_train['Transmission'] = X_train['Transmission'].apply({'Manual': 0, 'Automatic': 1}.get)
X_test['Transmission'] = X_test['Transmission'].apply({'Manual': 0, 'Automatic': 1}.get)


# In[ ]:


X_test.columns = map(str.lower, X_test.columns)


# In[ ]:


X_train.columns = map(str.lower, X_train.columns)


# In[ ]:


X_train.drop('power',  1, inplace=True)
X_test.drop('power',  1, inplace=True)


# In[ ]:


X_train.drop('engine',  1, inplace=True)
X_test.drop('engine',  1, inplace=True)


# In[ ]:


X_train.drop('mileage',  1, inplace=True)
X_test.drop('mileage',  1, inplace=True)


# In[ ]:


y_train


# In[ ]:


y_train = y_train * 0.014      ###conversion to US $


# In[ ]:


### save modified X and y  data frames
export_csv = df.to_csv (r'data\modified_train_set.csv', index = None, header=True) 


# ### Base Model

# In[ ]:


#Create Dummy Regression Always Predicts The Mean Value Of Target
# Create a dummy regressor
dummy_mean = DummyRegressor(strategy='mean')

# "Train" dummy regressor
dummy_mean.fit(X, y)


# In[ ]:


dummy_mean.predict(X)


# In[ ]:


# Get R-squared score
dummy_mean.score(X, y)  


# ### Making a model

# In[ ]:


X_train.head()


# In[ ]:


X_train.rename({"unnamed: 0":"a"}, axis="columns", inplace=True)
X_train.drop(["a"], axis=1, inplace=True)


# In[ ]:


X_test.rename({"unnamed: 0":"a"}, axis="columns", inplace=True)
X_test.drop(["a"], axis=1, inplace=True)


# In[ ]:


X_train


# In[ ]:


le = preprocessing.LabelEncoder()

le.fit( X_train['name'])
le.transform(X_train['name'])

le.fit( X_train['location'])
le.transform(X_train['location'])


# In[ ]:


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


# In[ ]:


Z_train= mapper.fit_transform(X_train)


# In[ ]:


Z_test = mapper.transform(X_test)


# In[ ]:


model = LinearRegression()
model.fit(Z_train, y_train)


# In[ ]:


model.predict(Z_test)


# In[ ]:


MSEs= cross_val_score(model,Z_train, y_train, cv=5  )


# In[ ]:


mean_MSE = np.mean(MSEs)
print(mean_MSE)


# #### Ridge with GridSearch

# In[ ]:


model = Ridge()

parameters = {
    'alpha' : [0.5, 1, 5, 10, 20],
    'max_iter' : [10, 50, 100], 
   }

gs = GridSearchCV(model, parameters, cv=5)

gs.fit(Z_train, y_train)


# In[ ]:


gs.best_score_


# In[ ]:


gs.best_params_


# In[ ]:


gs.best_estimator_


# ### CAT BOOST REGRESSOR

# In[ ]:


#cat_features = list(range(0, X_train.shape[1]))
#print(cat_features)


# In[ ]:


#print(f'Labels: {set(y)}')


# In[ ]:


cb_model = CatBoostRegressor(iterations=700,
                             learning_rate=0.05,
                             depth=5,
                             eval_metric='RMSE',
                             random_seed = 42)
                            
                             
                             
                            
                             


# In[ ]:


categorical_features_indices = np.where(X_train.dtypes != np.float)[0]


# In[ ]:


cb_model.fit(X_train, y_train,
             eval_set=(X_test,y_test),
             cat_features=categorical_features_indices,
             use_best_model=True,
             verbose=True)


# #####  make pipeline

# In[ ]:


cb_model.get_best_score()


# In[ ]:


pipe = make_pipeline(mapper, cb_model)


# In[ ]:


pipe.fit(X_train, y_train)


# In[ ]:


pipe.predict(X_test)


# ##### make a pickle

# In[ ]:


pickle.dump(pipe, open('model/pipe.pkl', 'wb'))
del pipe
pipe = pickle.load(open('model/pipe.pkl', 'rb'))
pipe


# In[ ]:




