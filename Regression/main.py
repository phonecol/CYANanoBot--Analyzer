import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pickle 
import pandas as pd 
import regressions
from sklearn.metrics import mean_squared_error


df = pd.read_csv("20ROI2_45min2021_10_18-10_54_07x.csv")
# print(color_data)
print(df.head())

features = ['R','G','B','H','Gray']
target = '# Cyanide Concentration'

X = df[features]
y = df[target]

X_test = [159.326543,  174.161235 , 192.469383 , 111.504198 , 173.762840]
X_test = [X_test]
print(X)
print(y)

# print("\nMULTIPLE LINEAR REGRESSION")
# regressions.multiple_linear_regression(X,y)

# print("\nMULTIPLE POLYNOMIAL REGRESSION")
# regressions.multiple_polynomial_regression(X,y)

# print("Advanced Linear Regression")
# regressions.advanced_linear_regression(X,y)

filename = 'multiple_linear_finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
pred = loaded_model.predict(X_test)
print(pred)
result = loaded_model.score(X, y)
print(result)