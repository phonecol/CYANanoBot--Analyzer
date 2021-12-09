from math import degrees
from matplotlib import colors
from dominantColors import DominantColors
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from imutils import build_montages
from operator import itemgetter
import time
import argparse

import plot_cn as pC
import pandas as pd
from Regression.regressions import multiple_linear_regression,multiple_polynomial_regression, linear_regression, polynomial_regression

# from openfolders_multiple import ROI_folder
# python create_regression_model.py -id2 000 -rf2 ROI2_newsensor_45min -ps new_sensor -rm Linear Regression -fr 'R','G','B'
ap = argparse.ArgumentParser()
ap.add_argument("-id2","--images_id", required=False,
    help ="images id")
ap.add_argument("-ps", "--paper_sensor",required=False,
    help="filename of the ROI2 folder")
ap.add_argument("-fr", "--feature",required=False,
    help="features")
ap.add_argument("-rm", "--regression_model",required=False,
    help="regression model")
ap.add_argument("-dr", "--degree",required=False, default=3, type= int,
    help="degree of the polynomial regression")

args = vars(ap.parse_args())
print(args)

regression_model = args['regression_model']
feature = args['feature']
degree = args['degree']


FILENAME_DIR = args['paper_sensor']
images_id_no = args['images_id']
# ROI2_DIR= args['ROI2_folder']
# fnamee =  ROI2_DIR

### LOCATE and INITIALIZE DIRECTORIES 
DATA_DIR = "DATA"
CD = os.getcwd()
print(CD)
backCD =os.path.normpath(os.getcwd() + os.sep + os.pardir)
print(backCD)
DATA_PATH = os.path.join(backCD,DATA_DIR)
print(DATA_PATH)
FILENAME_PATH = os.path.join(DATA_PATH,FILENAME_DIR)
print(FILENAME_DIR)
COLOR_DATA_DIR = "ColorData"
COLOR_DATA_PATH = os.path.join(FILENAME_PATH, COLOR_DATA_DIR)
data_dir = os.path.join(COLOR_DATA_PATH, str(images_id_no))
MODEL_DIR = "MODELS"
MODEL_PATH = os.path.join(data_dir, MODEL_DIR)

### CREATE DIRECTORY for MODELS
if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)
print(data_dir)





files = os.listdir(data_dir)
print(files)

R_2 = []
MSE = []

def find_csv_filenames( data_path, suffix=".csv" ):
    filenames = os.listdir(data_path)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

csv = find_csv_filenames(data_dir)
print(csv[-1])

csv_path = os.path.join(data_dir,csv[-1])
df = pd.read_csv(csv_path)
print(df.head())


###Group the 3 paper sensors and find the mean between them
# data = df.groupby(['# Cyanide Concentration']).mean()
# print(data.head())
# csv_path2 = os.path.join(data_dir,'averaged.csv')
# data.to_csv(csv_path2,index=True)
# print('Okay')
# df2 = pd.read_csv(csv_path2)
# print(df2.head)

try:
    input("Press ENTER to start the image acquisition")
except SyntaxError:
    pass

features = list(feature) #split the feature into characters
print(feature)
print(features)
target = '# Cyanide Concentration'

# X = df2[features]
# y = df2[target]

X = df[features]
y = df[target]

# print(X)
# print(y)
if regression_model == "Linear_Regression":
    print("\LINEAR REGRESSION")
    r_sq, mse = linear_regression(X,y,feature, MODEL_PATH)
    R_2.append(r_sq)
    MSE.append(mse)


elif regression_model == "Multiple_Linear_Regression":
    print("\MULTIPLE LINEAR REGRESSION")
    r_sq, mse = multiple_linear_regression(X,y,feature,MODEL_PATH)
    R_2.append(r_sq)
    MSE.append(mse)

elif regression_model == "Multiple_Polynomial_Regression":
    print("\MULTIPLE POLYNOMIAL REGRESSION")
    r_sq, mse = multiple_polynomial_regression(X,y,feature,degree,MODEL_PATH)
    R_2.append(r_sq)
    MSE.append(mse)



elif regression_model == "Polynomial_Regression":
    print("\POLYNOMIAL REGRESSION")
    r_sq, mse = polynomial_regression(X,y,feature,degree ,MODEL_PATH)
    R_2.append(r_sq)
    MSE.append(mse)

