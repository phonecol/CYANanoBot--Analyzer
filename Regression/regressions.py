import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pickle 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
### Function for simple linear regression
def linear_regression(x,y,filename_path):

    x_train, x_test,y_train,y_test = train_test_split(x,y,test_size =0.2)
    #create a model and fit it
    model = LinearRegression()
    model.fit(x_train,y_train)

    #or pde ra ingani
    #model = LinearRegression().fit(x,y)

    #get results
    r_sq = model.score(x_test,y_test)
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)


    ##predict response 
    y_pred = model.predict(x_test)
    print('predicted response:', y_pred, sep='\n')
    print('actual response:', y_test, sep='\n')

    #calculate mean squared error 
    mse = mean_squared_error(y_test, y_pred, squared=False)
    print("Mean Squared Error: ",mse)

    #save model
    filename = filename_path+'\\linear_finalized_model_'+ str(mse)+'.sav'
    pickle.dump(model, open(filename, 'wb'))

    return r_sq,mse
 
# some time later...
 
# load the model from disk


### Function for multiple linear regression
def multiple_linear_regression(x,y,filename_path):

    x_train, x_test,y_train,y_test = train_test_split(x,y,test_size =0.2)
    
    #create a model and fit it 
    model = LinearRegression().fit(x_train,y_train)


    #get results
    r_sq = model.score(x,y)
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)

    #predict response 
    y_pred = model.predict(x_test)
    print('predicted response:', y_pred, sep='\n')
    print('actual response:', y_test, sep='\n')
    print(y_pred-y_test)
    #calculate mean squared error 
    mse = mean_squared_error(y_test, y_pred, squared=False)
    print("Mean Squared Error: ",mse)

    #save model
    filename = filename_path+'\\multiple_linear_finalized_model_'+ str(mse)+'.sav'
    pickle.dump(model, open(filename, 'wb'))
    
    return r_sq,mse
 

### Function for polynomial regression

def polynomial_regression(x,y,filename_path):

    x_train, x_test,y_train,y_test = train_test_split(x,y,test_size =0.2)
    #transform input data 
    transformer_train = PolynomialFeatures(degree=2, include_bias=False)
    transformer_train.fit(x_train)
    x_train_ = transformer_train.transform(x_train)
    
    transformer_test = PolynomialFeatures(degree=2, include_bias=False)
    transformer_test.fit(x_test)
    x_test_ = transformer_test.transform(x_test)
    
    #or pde ingani 
    #x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)
    # print(x_)

    #create a model and fit it
    model = LinearRegression().fit(x_train_,y_train)


    #get results 
    r_sq = model.score(x_test_, y_test)
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    print('coefficients:', model.coef_)

    #predict response
    y_pred = model.predict(x_test_)
    print('predicted response:', y_pred, sep='\n')
    print('actual response:', y_test, sep='\n')

    #calculate mean squared error 
    mse = mean_squared_error(y_test, y_pred, squared=False)
    print("Mean Squared Error: ",mse)

    #save model
    filename = filename_path+'\\polynomial_finalized_model_'+ str(mse)+'.sav'
    pickle.dump(model, open(filename, 'wb'))
    # return x_

    return r_sq,mse
 
def multiple_polynomial_regression(x,y,filename_path):
    # Step 2b: Transform input data
    x_train, x_test,y_train,y_test = train_test_split(x,y,test_size =0.2)
    #transform input data 
    transformer_train = PolynomialFeatures(degree=2, include_bias=False)
    transformer_train.fit(x_train)
    x_train_ = transformer_train.transform(x_train)
    
    transformer_test = PolynomialFeatures(degree=2, include_bias=False)
    transformer_test.fit(x_test)
    x_test_ = transformer_test.transform(x_test)

    # Step 3: Create a model and fit it
    model = LinearRegression().fit(x_train_, y_train)

    # Step 4: Get results
    r_sq = model.score(x_test_, y_test)
    intercept, coefficients = model.intercept_, model.coef_
    print('coefficient of determination:', r_sq)
    print('intercept:', intercept)
    print('coefficients:', coefficients, sep='\n')

    # Step 5: Predict
    y_pred = model.predict(x_test_)
    
    print('predicted response:', y_pred, sep='\n')
    print('actual response:', y_test, sep='\n')
    
    #calculate mean squared error 
    mse = mean_squared_error(y_test, y_pred, squared=False)
    print("Mean Squared Error: ",mse)

    #save model
    
    filename = filename_path+'\\multiple_polynomial_finalized_model_'+ str(mse)+'.sav'
    pickle.dump(model, open(filename, 'wb'))


    return r_sq,mse
    
    # return x_

def advanced_linear_regression(x,y):
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


# x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
# y = np.array([5, 20, 14, 32, 22, 38])

# print("Simple Linear Regression")
# linear_regression(x,y)
# filename = 'linear_finalized_model.sav'
# loaded_model = pickle.load(open(filename, 'rb'))
# pred = loaded_model.predict(x)
# print(pred)
# result = loaded_model.score(x, y)
# print(result)

# print("Done")




# x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
# y = [4, 5, 20, 14, 32, 22, 38, 43]
# x, y = np.array(x), np.array(y)

# print("Multiple Linear Regression")
# multiple_linear_regression(x,y)

# filename = 'multiple_linear_finalized_model.sav'
# loaded_model = pickle.load(open(filename, 'rb'))
# pred = loaded_model.predict(x)
# print(pred)
# result = loaded_model.score(x, y)
# print(result)
# print("Done")

# x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
# y = [4, 5, 20, 14, 32, 22, 38, 43]
# x, y = np.array(x), np.array(y)

# print("Multiple Polynomial Regression")
# x_= multiple_polynomial_regression(x,y)
# filename = 'multiple_polynomial_finalized_model.sav'
# loaded_model = pickle.load(open(filename, 'rb'))
# pred=loaded_model.predict(x_)

# result = loaded_model.score(x_, y)
# print(pred)
# print(result)
# print("Done")

# x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
# y = np.array([15, 11, 2, 8, 25, 32])

# print("Polynomial Regression")
# x_= polynomial_regression(x,y)

# filename = 'polynomial_finalized_model.sav'
# loaded_model = pickle.load(open(filename, 'rb'))
# pred = loaded_model.predict(x_)
# print(pred)
# result = loaded_model.score(x_, y)

# print(result)
# print("Done")




