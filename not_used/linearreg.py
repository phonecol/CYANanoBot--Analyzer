print(__doc__)


# Code source: Jaques Grobler
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split
import pandas as pd
# Load the diabetes dataset

colorData = pd.read_csv("ColorData/000_ColorSortedData_2021_09_16-11_48_33.csv")
print(colorData)
X = colorData.iloc[:,0].values.reshape(-1, 1)

y = colorData.iloc[:,1].values.reshape(-1, 1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(X_train)
print(X)
print(y)

model = LinearRegression()
model.fit(X,y)

r_sq = model.score(X_train,y_train)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
y_pred = model.intercept_ + model.coef_ * X_test
yhat = model.predict(X_test)
# print('predicted response:', y_pred, sep='\n')

plt.scatter(X, y)
plt.plot(X_test, yhat, color='red')
plt.plot(X_test, y_pred, color='black')
plt.show()