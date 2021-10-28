import numpy as np
from sklearn.linear_model import LinearRegression


#provide data
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])


def linear_regression(x,y):

#create a model and fit it
    model = LinearRegression()
    model.fit(x,y)

    #or pde ra ingani
    #model = LinearRegression().fit(x,y)

    #get results
    r_sq = model.score(x,y)
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)


    ##predict response 
    y_pred = model.predict(x)
    print('predicted response:', y_pred, sep='\n')