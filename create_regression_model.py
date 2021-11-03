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
# from openfolders_multiple import FILENAME_DIRECTORY
import plot_cn as pC
import pandas as pd
from Regression.regressions import multiple_linear_regression

# from openfolders_multiple import ROI_folder
# python color_extraction.py -id2 000 -rf2 ROI2_newsensor_45min
ap = argparse.ArgumentParser()
ap.add_argument("-id2","--images_id", required=False,
    help ="images id")
ap.add_argument("-fn", "--filename",required=False,
    help="filename of the ROI2 folder")
ap.add_argument("-rf2", "--ROI2_folder",required=False,
    help="filename of the ROI2 folder")

args = vars(ap.parse_args())
print(args)

filename = args['filename']
images_id_no = args['images_id']
ROI2_folder= args['ROI2_folder']
fnamee =  ROI2_folder


CD = os.getcwd()
print(CD)

backCD =os.path.normpath(os.getcwd() + os.sep + os.pardir)
print(backCD)

DATA_DIRECTORY = os.path.join(backCD,"DATA")
print(DATA_DIRECTORY)

FILENAME_DIRECTORY = os.path.join(DATA_DIRECTORY,filename)
print(FILENAME_DIRECTORY)

color_data_folder = os.path.join(FILENAME_DIRECTORY, "ColorData")

data_path = os.path.join(color_data_folder, str(images_id_no))

print(data_path)

files = os.listdir(data_path)
print(files)


def find_csv_filenames( data_path, suffix=".csv" ):
    filenames = os.listdir(data_path)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

csv = find_csv_filenames(data_path)
print(csv[-1])

csv_path = os.path.join(data_path,csv[-1])
df = pd.read_csv(csv_path)
print(df.head())


features = ['R','G','B','H']
target = '# Cyanide Concentration'

X = df[features]
y = df[target]

X_test = [159.326543,  174.161235 , 192.469383 , 111.504198 , 173.762840]
X_test = [X_test]
print(X)
print(y)

print("\nMULTIPLE LINEAR REGRESSION")
multiple_linear_regression(X,y,FILENAME_DIRECTORY)