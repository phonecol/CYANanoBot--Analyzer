import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#provide data
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([15, 11, 2, 8, 25, 32])

#transform input data 
def polynomial_regression(x,y):
    #transform input data 
    transformer = PolynomialFeatures(degree=2, include_bias=False)
    transformer.fit(x)
    x_ = transformer.transform(x)
    #or pde ingani 
    #x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)
    print(x_)

    #create a model and fit it
    model = LinearRegression().fit(x_,y)


    #get results 
    r_sq = model.score(x_, y)
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    print('coefficients:', model.coef_)

    #predict response
    y_pred = model.predict(x_)
    print('predicted response:', y_pred, sep='\n')