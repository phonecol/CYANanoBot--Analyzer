import numpy as np
import statsmodels.api as sm

x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)

def advanced_linear_regression():
    x = sm.add_constant(x)
    print(x)
    print(y)

    #create a model and fit it

    model = sm.OLS(y,x)
    results = model.fit()

    #get results
    print(results.summary())

    #predict response 
    print('predicted response:', results.fittedvalues, sep='\n')
    print('predicted response:', results.predict(x), sep='\n')


