import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
# Load the diabetes dataset

colorData = pd.read_csv("ColorData/000_ColorSortedData_2021_09_16-11_48_33.csv")
print(colorData)
cColorData = colorData[['# Cyanide Concentration','R','G','B']]
cColorData.head(9)
X = cColorData.iloc[:,0].values.reshape(-1, 1)

y = cColorData.iloc[:,1].values.reshape(-1, 1)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
plt.scatter(X, y,  color='blue')
plt.xlabel("Cyanide Concentration")
plt.ylabel("Mean Pixel Intensity")
plt.title("Red")
plt.show()

msk = np.random.rand(len(colorData)) < 0.8
train = colorData[msk]
test = colorData[~msk]

train_x = np.asanyarray(train[['# Cyanide Concentration']])
train_y = np.asanyarray(train[['R']])

test_x = np.asanyarray(test[['# Cyanide Concentration']])
test_y = np.asanyarray(test[['R']])


poly = PolynomialFeatures(degree =2)
train_x_ = poly.fit_transform(train_x)
train_x_

model = linear_model.LinearRegression()
train_y_ = model.fit(train_x_, train_y)

# The coefficients
print ('Coefficients: ', model.coef_)
print ('Intercept: ',model.intercept_)


plt.scatter(train_x, train_y,  color='blue')
XX = np.arange(0.0, 120.0, 10)
yy = model.intercept_[0]+ model.coef_[0][1]*XX+ model.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, '-r' )
plt.xlabel("Cyanide Concentration")
plt.ylabel("Mean Pixel Intensity")
plt.title("Red")
plt.show()