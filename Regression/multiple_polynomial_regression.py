# Step 1: Import packages
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Step 2a: Provide data
x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)


def multiple_polynomial_regression(x,y):
    # Step 2b: Transform input data
    x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)

    # Step 3: Create a model and fit it
    model = LinearRegression().fit(x_, y)

    # Step 4: Get results
    r_sq = model.score(x_, y)
    intercept, coefficients = model.intercept_, model.coef_

    # Step 5: Predict
    y_pred = model.predict(x_)


    print('coefficient of determination:', r_sq)
    print('intercept:', intercept)
    print('coefficients:', coefficients, sep='\n')
    print('predicted response:', y_pred, sep='\n')


multiple_polynomial_regression(x,y)
print("done")