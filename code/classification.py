#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

def feature_imp(df,model):
    fi = pd.DataFrame()
    fi["feature"] = df.columns
    fi["importance"] = model.feature_importances_
    return fi.sort_values(by="importance", ascending=False)

#plot opens in seperate window
#%matplotlib qt

#inline plot
#%matplotlib inline

#dict for dtypes
dtypes = {'gender':'category', 'diabetes':'category','hypertension':'category',
          'stroke':'category' ,'heart disease':'int64'}

#big city health data
df_data = pd.read_csv('/home/ubunt/Schreibtisch/Studium/8_Semester/MedizinischeInformatik/Projekt/MQIC_Patient/data/MQIC_Patient_Data_100k_Sample.csv',
                      dtype=dtypes)

#data cleaning
# TODO:  maybe also drop 'Other' in gender-col
print('Data cleaning...')
entries_before = df_data.shape[0]
df_data.dropna(axis=0, how='any', inplace=True)
entries_after = df_data.shape[0]
dropped = entries_before - entries_after
print(f'{dropped} of {entries_before} lines dropped')

#change dtype
# TODO: add to dict above
df_data['smoking history'] = df_data['smoking history'].astype('category')

'''
# plot
# TODO: open plot in own window
df_data['gender'].value_counts().plot(kind='bar')
df_data['age'].plot(kind='box')
df_data['diabetes'].value_counts().plot(kind='bar')
df_data['hypertension'].value_counts().plot(kind='bar')
df_data['stroke'].value_counts().plot(kind='bar')
df_data['heart disease'].value_counts().plot(kind='bar')
df_data['smoking history'].value_counts().plot(kind='bar')
df_data['BMI'].plot(kind='box')
'''

#rearrange df
df_data = df_data[['gender','age','diabetes','hypertension','stroke',
                   'smoking history','BMI', 'heart disease']]
#input variables X
X = df_data[['gender','age','diabetes','hypertension','stroke',
                   'smoking history','BMI']]
#output variable y
y = df_data['heart disease']




# Data Prep
categorial = df_data[['gender', 'diabetes', 'hypertension', 'stroke', 'smoking history']].values


#onehotencode categorical data, i.e. every col except age and BMI

# TODO: maybe drop one hot encoding. Is one hot encoding necessary?
#one hot encode  categorical cols (see categorial) --> prep for xgboost
encoded_X = None
for i in range(0, categorial.shape[1]):
    label_encoder = LabelEncoder()
    feature = label_encoder.fit_transform(categorial[:,i])
    feature = feature.reshape(categorial.shape[0], 1)
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    feature = onehot_encoder.fit_transform(feature)
    #add feature to encoded_X
    if encoded_X is None:
        encoded_X = feature
    else:
        encoded_X = np.concatenate((encoded_X, feature), axis=1)


#add 'age','BMI'
numeric_cols = ['age', 'BMI']
for col in numeric_cols:
    values = X[col].values
    values = values.reshape(values.shape[0],1)
    encoded_X = np.concatenate((encoded_X, values), axis=1)

#encoded_X col labels: 
#TODO: label e.g. gender_0 to gender_male and so forth    
labels = ['gender_0', 'gender_1', 'gender_2', 'diabetes_0', 'diabetes_1',
        'hypertension_0','hypertension_1', 'stroke_0', 'stroke_1', 
        'smoking history_0', 'smoking history_1', 'smoking history_2', 
        'smoking history_3', 'smoking history_4','age', 'BMI']  

#reshape output variables
    
reshaped_y = y.values
reshaped_y = reshaped_y.reshape(y.shape[0],1) 
     
#split train and test data

X_train, X_test, y_train, y_test = train_test_split(encoded_X, reshaped_y, test_size = 0.2)
###############################################################################
# Model
###############################################################################

#fit model
model = XGBClassifier()
model.fit(X_train, y_train)
print(model)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


###############################################################################
#show feature importance
# TODO: sum groups together e.g. diabetes_0 and diabetes_1
df_enc = pd.DataFrame(encoded_X)
df_enc.columns = labels
featimp_temp = feature_imp(df_enc, model)
featimp_temp.set_index('feature', inplace=True)
#sum up features belonging together
featimp = pd.DataFrame()
featimp['age'] = featimp_temp.loc['age']
featimp['BMI'] = featimp_temp.loc['BMI']
featimp['gender'] = featimp_temp.loc[['gender_0', 'gender_1', 'gender_2']].sum()
featimp['diabetes'] = featimp_temp.loc[['diabetes_0', 'diabetes_1']].sum()
featimp['hypertension'] = featimp_temp.loc[['hypertension_0','hypertension_1']].sum()
featimp['stroke'] = featimp_temp.loc[['stroke_0', 'stroke_1']].sum()
featimp['smoking history'] = featimp_temp.loc[['smoking history_0', 'smoking history_1', 'smoking history_2', 
        'smoking history_3', 'smoking history_4']].sum()
featimp.reset_index()
featimp['feature'] = featimp.columns

featimp = pd.DataFrame()
featimp['feature'] = [['gender','age','diabetes','hypertension','stroke',
                   'smoking history','BMI']]

featimp_temp.plot('feature', 'importance', 'barh', figsize=(12,7), legend=False)

