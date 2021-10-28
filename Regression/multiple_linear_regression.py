import numpy as np
from sklearn.linear_model import LinearRegression

x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
print(x)
print(y)

x, y = np.array(x), np.array(y)
print(x)
print(y)

def multiple_linear_regression(x,y):
    #create a model and fit it 
    model = LinearRegression().fit(x,y)


    #get results
    r_sq = model.score(x,y)
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)

    #predict response 
    y_pred = model.predict(x)
    print('predicted response:', y_pred, sep='\n')


