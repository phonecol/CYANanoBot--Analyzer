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
from paper_sensor_detection_program import DATA_DIR
# from openfolders_multiple import FILENAME_DIRECTORY
import plot_cn as pC
import pandas as pd
from Regression.regressions import multiple_linear_regression,multiple_polynomial_regression

# from openfolders_multiple import ROI_folder
# python create_regression_model.py -id2 000 -rf2 ROI2_newsensor_45min -fn new_sensor
ap = argparse.ArgumentParser()
ap.add_argument("-id2","--images_id", required=False,
    help ="images id")
ap.add_argument("-fn", "--filename",required=False,
    help="filename of the ROI2 folder")
ap.add_argument("-rf2", "--ROI2_folder",required=False,
    help="filename of the ROI2 folder")

args = vars(ap.parse_args())
print(args)

FILENAME_DIR = args['filename']
images_id_no = args['images_id']
ROI2_DIR= args['ROI2_folder']
fnamee =  ROI2_DIR
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


features = ['R','G','B']
target = '# Cyanide Concentration'

X = df[features]
y = df[target]


# print(X)
# print(y)

print("\nMULTIPLE LINEAR REGRESSION")
r_sq, mse = multiple_linear_regression(X,y,data_dir)
R_2.append(r_sq)
MSE.append(mse)


features_1 = ['R','G']
target_1 = '# Cyanide Concentration'

X_1 = df[features_1]
y_1 = df[target_1]


# print(X_1)
# print(y_1)

print("\nMULTIPLE LINEAR REGRESSION")
r_sq, mse =multiple_linear_regression(X_1,y_1,data_dir)
R_2.append(r_sq)
MSE.append(mse)


features_2 = ['R','B']
target_2 = '# Cyanide Concentration'

X_2 = df[features_2]
y_2 = df[target_2]


# print(X_2)
# print(y_2)

print("\nMULTIPLE LINEAR REGRESSION")
r_sq, mse =multiple_linear_regression(X_2,y_2,data_dir)
R_2.append(r_sq)
MSE.append(mse)

features_3 = ['B','G']
target_3 = '# Cyanide Concentration'

X_3 = df[features_3]
y_3 = df[target_3]


# print(X_3)
# print(y_3)

print("\nMULTIPLE LINEAR REGRESSION")
r_sq, mse =multiple_linear_regression(X_3,y_3,data_dir)
R_2.append(r_sq)
MSE.append(mse)

print("Coefficients of Determination: ",R_2)
print("Mean Squared Errors: ", MSE)