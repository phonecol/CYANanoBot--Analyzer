import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pickle 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils.extmath import make_nonnegative
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import pandas as pd


### Function for simple linear regression
def linear_regression(x,y,feature,filename_path):

    # x_train, x_test,y_train,y_test = train_test_split(x,y,test_size =0.2)
    x_train = x
    x_test = x
    y_train =y
    y_test =y
    # #create a model and fit it
    model = LinearRegression()
    model.fit(x_train,y_train)

    #or pde ra ingani
    #model = LinearRegression().fit(x,y)

    #get results
    r_sq = model.score(x_test,y_test)
    r_sq= round(r_sq, 4)
    
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)


    ##predict response 
    y_pred = model.predict(x_test)
    
    y_pred = pd.Series(y_pred, name='Prediction')
    
    r2 = r2_score(y_test, y_pred)
    print("R2", r2)

    #calculate prediction - actual
    print(y_pred-y_test)
    prediction_error = y_test - y_pred
    combined = pd.concat([y_pred, y_test, prediction_error], axis=1)
    combined.columns =['Predicted', 'Actual', 'Prediction Error']
    print(combined)
    #calculate mean squared error 
    mse = mean_squared_error(y_test, y_pred, squared=False)
    
    mse = round(mse,4)
    print("Root Mean Squared Error: ",mse)

    #save model
    filename = filename_path+'\\linear_finalized_model_'+ feature+'_'+str(mse)+'_'+str(r_sq)+'.sav'
    pickle.dump(model, open(filename, 'wb'))
    xfit = np.linspace(0, 250, 1000)
    yfit = model.predict(xfit[:, np.newaxis])

    plt.scatter(x_test, y_test,color='g')
    plt.plot( x_test,y_pred,color='r')
    plt.plot(xfit,yfit,color= 'y')
    # plt.legend("R_sq: %s",r_sq)
    str_r2 = "R^2 = " + str(r_sq)
    str_mse = "RMSE = " +str(mse)
    plt.text(0, 120, str_r2)
    plt.title("{} Channel" .format(feature))
    plt.text(0, 110, str_mse)
    plt.ylim(-10, 130)
    plt.xlim(0, 270)
    plt.ylabel("Cyanide Concentration (PPM)")
    plt.xlabel("Mean Pixel Intensity")
    
    plt.savefig(filename_path+'/linear_finalized_model_'+feature+'_' +str(mse)+'_'+str(r_sq)+'.png')
    plt.show()

    return r_sq,mse
 




### Function for multiple linear regression
def multiple_linear_regression(x,y,feature,filename_path):

    # x_train, x_test,y_train,y_test = train_test_split(x,y,test_size =0.2)
    x_train = x
    x_test = x
    y_train =y
    y_test =y

    # #create a model and fit it 
    model = LinearRegression().fit(x_train,y_train)


    #get results
    r_sq = model.score(x,y)
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)

    #predict response 
    y_pred = model.predict(x_test)
    y_pred = pd.Series(y_pred)
    
    #calculate prediction - actual
    prediction_error =abs( y_test - y_pred)
    combined = pd.concat([y_pred, y_test, prediction_error], axis=1)
    combined.columns =['Predicted', 'Actual', 'Prediction Error']
    print(combined)
    #calculate mean squared error 
    mse = mean_squared_error(y_test, y_pred, squared=False)
    r_sq= round(r_sq, 4)
    mse = round(mse,4)
    print("Root Mean Squared Error: ",mse)

    #save model
    filename = filename_path+'\\multiple_linear_finalized_model_'+feature+'_'+ str(mse)+'_'+str(r_sq)+'.sav'
    pickle.dump(model, open(filename, 'wb'))
    
    return r_sq,mse
 

### Function for polynomial regression

def polynomial_regression(x,y,feature, degree,filename_path):

    # x_train, x_test,y_train,y_test = train_test_split(x,y,test_size =0.2)
    #transform input data 
    x_train = x
    x_test = x
    y_train =y
    y_test =y
    polyreg = make_pipeline(PolynomialFeatures(degree, include_bias=False),LinearRegression(fit_intercept=False))


    model = polyreg.fit(x_train,y_train)

    #get results 
    r_sq = model.score(x_test, y_test)
    print('coefficient of determination:', r_sq)
    print('intercept:', model.named_steps.linearregression.intercept_)
    print('coefficients:', model.named_steps.linearregression.coef_)
    
    #predict response
    y_pred = model.predict(x_test)
    y_pred = pd.Series(y_pred)
    r2 = r2_score(y_test, y_pred)
    print("R2", r2)

    #calculate prediction - actual
    print(y_pred-y_test)
    prediction_error =abs( y_test - y_pred)
    combined = pd.concat([y_pred, y_test, prediction_error], axis=1)
    combined.columns =['Predicted', 'Actual', 'Prediction Error']
    print(combined)

    #calculate mean squared error 
    mse = mean_squared_error(y_test, y_pred, squared=False) #if squared = False: RMSE #### if true: MSE
    r_sq= round(r_sq, 4)
    mse = round(mse,4)
    print("Root Mean Squared Error: ",mse)

    #save model
    filename = filename_path+'\\polynomial_finalized_model_'+feature+'_'+ str(mse)+'_'+str(r_sq)+'.sav'
    pickle.dump(model, open(filename, 'wb'))

    ####Plot##########
    xfit = np.linspace(0, 250, 1000)
    # yfit = model.predict(xfit[:, np.newaxis])
    plt.plot(x_test, model.predict(x_test), color = 'red')

    plt.scatter(x_test, y_test,color='g')
    plt.plot( x_test,y_pred,color='r')
    # plt.plot(yfit,xfit,color= 'y')
    # plt.legend("R_sq: %s",r_sq)
    str_r2 = "R^2 = " + str(r_sq)
    str_mse = "RMSE = " +str(mse)
    plt.text(0, 110, str_r2)
    plt.title("{} channel".format(feature))
    plt.text(0, 100, str_mse)
    plt.ylim(-10, 130)
    plt.xlim(0, 270)
    plt.ylabel("Cyanide Concentration (PPM)")
    plt.xlabel("Mean Pixel Intensity")
    plt.savefig(filename_path+'/polynomial_finalized_model_'+ feature+'_'+str(mse)+'_'+str(r_sq)+'.png')
    plt.show()
    
    # return x_

    return r_sq,mse
 
def multiple_polynomial_regression(x,y,feature,degree,filename_path):
    # Step 2b: Transform input data
    # x_train, x_test,y_train,y_test = train_test_split(x,y,test_size =0.2)
    x_train = x
    x_test = x
    y_train =y
    y_test =y
    #transform input data 
    polyreg = make_pipeline(PolynomialFeatures(degree, include_bias= False),LinearRegression(fit_intercept=False))
    model = polyreg.fit(x_train,y_train)
    # Step 4: Get results
    r_sq = model.score(x_test, y_test)
    print('intercept:', model.named_steps.linearregression.intercept_)
    print('coefficients:', model.named_steps.linearregression.coef_)
    print('coefficient of determination:', r_sq)

    # Step 5: Predict
    y_pred = model.predict(x_test)
    y_pred = pd.Series(y_pred)

    #calculate prediction - actual
    prediction_error =abs( y_test - y_pred)
    combined = pd.concat([y_pred, y_test, prediction_error], axis=1)
    combined.columns =['Predicted', 'Actual', 'Prediction Error']
    print(combined)

    #calculate mean squared error 
    mse = mean_squared_error(y_test, y_pred, squared=False)
    print("Root Mean Squared Error: ",mse)
    r_sq= round(r_sq, 4)
    mse = round(mse,4)

    #save model
    filename = filename_path+'\\multiple_polynomial_finalized_model_'+feature+'_'+ str(mse)+'_'+str(r_sq)+'.sav'
    pickle.dump(model, open(filename, 'wb'))
   

    return r_sq,mse
    

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


