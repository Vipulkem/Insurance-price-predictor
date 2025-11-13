import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score
import warnings
warnings.filterwarnings('ignore')


st.title("Insurance price Predictor")

Insurance_Data = pd.read_csv('insurance.csv')


# Data Preprocessing

categorical_features = []
Numeric_features = []

for column in Insurance_Data.columns:
    if Insurance_Data[column].dtype == 'object':
        categorical_features.append(column)
    else:
        Numeric_features.append(column)

Insurance_Data['sex'] = np.where(Insurance_Data['sex'] == 'male', 1, 0)
Insurance_Data['smoker'] = np.where(Insurance_Data['smoker'] == 'yes', 1, 0)
Insurance_Data.replace({'region': {'southwest':0, 'southeast':1, 'northwest':2, 'northeast':3}}, inplace=True)

# Spiliting the features and Target
X = Insurance_Data.drop('charges', axis=1)
Y = Insurance_Data['charges']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)   


# Model Training and Fitting

Linear_model = LinearRegression()

Linear_model.fit(X_train, Y_train)

# Model Prediction

traning_pred = Linear_model.predict(X_train)

# R Squared value
r2_train = r2_score(Y_train, traning_pred)
print("R Squared value for training data: ", r2_train)

# prediction on test data
test_pred = Linear_model.predict(X_test)

# R Squared value
r2_test = r2_score(Y_test, test_pred)
print('R Squared value for test data: ', r2_test)

print(r2_train-r2_test)

with open('insurance_model.pkl', 'wb') as file:
    pickle.dump(Linear_model, file)