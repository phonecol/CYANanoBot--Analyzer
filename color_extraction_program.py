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

# from openfolders_multiple import ROI_folder
# python color_extraction.py -id2 000 -rf2 ROI2_newsensor_45min -ps paper_sensor_name
ap = argparse.ArgumentParser()
ap.add_argument("-id2","--images_id", required=False,
    help ="images id")
ap.add_argument("-ps", "--paper_sensor",required=False,
    help="filename of the paper sensor folder")

args = vars(ap.parse_args())
print(args)

FILENAME_DIR = args['paper_sensor']
images_id_no = args['images_id']
ROI2_folder= 'ROI2'
fnamee =  ROI2_folder
DATA_DIR = "DATA"



###DATA
######PAPER_SENSOR
#########model
#########ROI_PAPER_SENSOR
#########ROI2_PAPER_SENSOR
#########ColorData
############IMG_ID
###############CSV
###############PLOTS
###############HISTOGRAMS
##################HistogramsCSV
##################HistogramsPNG

CD = os.getcwd()
print(CD)

backCD =os.path.normpath(os.getcwd() + os.sep + os.pardir)
print(backCD)

DATA_PATH = os.path.join(backCD,DATA_DIR)
print(DATA_PATH)

FILENAME_PATH = os.path.join(DATA_PATH,FILENAME_DIR)
print(FILENAME_PATH)

if not os.path.exists(FILENAME_PATH):
    os.mkdir(FILENAME_PATH)

ROI2_PATH = os.path.join(FILENAME_PATH,str(ROI2_folder))

if not os.path.exists(ROI2_PATH):
    os.mkdir(ROI2_PATH)

IMAGE_PATH = os.path.join(ROI2_PATH,str(images_id_no))

if not os.path.exists(IMAGE_PATH):
    os.mkdir(IMAGE_PATH)

print(IMAGE_PATH)

COLOR_DATA_DIR = "ColorData"
COLOR_DATA_PATH = os.path.join(FILENAME_PATH, COLOR_DATA_DIR)
print(COLOR_DATA_PATH)
if not os.path.exists(COLOR_DATA_PATH):
    os.mkdir(COLOR_DATA_PATH)

# color_data_folder = os.path.join(color_data_folder, ROI2_folder)
# print(color_data_folder)

data_path = os.path.join(COLOR_DATA_PATH, str(images_id_no))
print(data_path)

if not os.path.exists(data_path):
    os.mkdir(data_path)

histograms_dir = "Histograms"
histograms_path = os.path.join(data_path,histograms_dir)
if not os.path.exists(histograms_path):
    os.mkdir(histograms_path)

coord_no = "8"



# coord_noo= "3"
#000 first image without water
#00 first image with water
#01 2nd image with water after 30sec
#02 3rd image with water after 1min

# Function for reading an image file
def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return image


#The filenames of the images of the ROI of paper sensor must be in this format the [coords,cyanide_concentration] e.g.: 1,90ppm 
#the directory of the images of the paper sensor


def get_images_from_a_folder(path, coord_no):
#initialize the list of images, and its filenames
    images = []
    ppm_values = []
    combined = []
    cn_Concentrations=[]
    coords = []
    print(path)
    files = os.listdir(path)
    files = natsorted(files)
    print(files)

    #a loop for getting the images in the folder and append them into a list
    for file in files:
        image = get_image(os.path.join(path, file))
        # ppm_value1, image_type = file.split('.')#split the filename to remove the image type
        ppm_value1 = file[:-4]
        coord, cn_Concentration = ppm_value1.split(',')#split the coordinate of the paper sensor in the spotplate and the cyanide concentration
        cn_Concentration = cn_Concentration[:-3]

        #append the images, cyanide concentrations, and coordinates
        # if coord == coord_no:
        images.append(image)
        ppm_values.append(cn_Concentration)
        combined.append((image, cn_Concentration))
        cn_Concentrations.append(cn_Concentration)
        coords.append(coord)

    return images, ppm_values, cn_Concentrations, coords

#function for saving the data
def save_data(data,image_number,data_path, timestr,fname):
    data = data.T ##transpose the data array##
    sorted_data = natsorted(data,key=itemgetter(0))##sort the data by their coordinates##
    header = 'Cyanide Concentration,coordinate,R,G,B,R_std,G_std,B_std,H,S,V,H_std,S_std,V_std,L,a,b,L_std,a_std,b_std,Gray,Gray_std' #initialize the header for the csv file
    filename_Data =data_path+"\\"+image_number+fname + timestr+".csv" ##initialize the filename of the data
    filename_Sorted_Data =data_path+"\\"+image_number+fname + timestr+"x.csv"  ##initialize the filename of the sorted data
    data = np.array(data)## convert the data and sorted data into numpy arrays
    sorted_data = np.array(sorted_data)## convert the data and sorted data into numpy arrays
    np.savetxt(filename_Data, data, delimiter=",",header= header,fmt='%s') #save the data array in a csv filetype with a filename of "data.csv" with the following header defined above
    np.savetxt(filename_Sorted_Data, sorted_data, delimiter=",",header= header,fmt='%s') #save the data array in a csv filetype with a filename of "data.csv" with the following header defined above
    print("ColorData has been saved")
    return  data, sorted_data



def main():
    #for KMeans Algorithm
    clusters = 3 #number of clusters of colors. Value is at the range of (2-5)
    index = 1
    coords = []
    cn_Concentrations = []
    RGB_KMeans = []
    RGB_Means = []
    RGB_stds =[]
    HSV_Means = []
    HSV_stds =[]
    Lab_Means = []
    Lab_stds =[]
    Gray_Means = []
    Gray_stds = []
    colorspaces = []
    tail = os.path.split(IMAGE_PATH)
    image_number = tail[1]
    images , ppm_values, cn_Concentrations, coords = get_images_from_a_folder(IMAGE_PATH, coord_no)
    timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
    print(timestr)
    # print(len(images))

    for i in range(len(images)):

        dc = DominantColors(images[i],images,clusters) #initialize the DominantColors class
        dc.saveHistogram(histograms_path+"\\{}Histogram".format(i), False) #set to false para dili i show ang histogram plot
        rgb_kmeans = dc.dominantColors()  #call the dominantColors function to get the dominant colors of the image using KMeans Algorithm
        rgb_kmeans = rgb_kmeans.flatten()
        print("Dominant Colors: ",rgb_kmeans)
        print("Dominant Colors sorted: ",rgb_kmeans)
        hsv,lab,gray = dc.cvtColorSpace()
        # dc.plotHistogram()
        # dc.colorPixels()
        rgb_mean,rgb_std,hsv_mean,hsv_std,lab_mean,lab_std, gray_mean, gray_std = dc.getMeanIntensity()  #call the getMeanIntensity function to get the average RGB pixel intensity and its standard deviation of the paper sensor
        #append the RGB, HSV,Lab, Gray Values into a list
        RGB_KMeans.append(rgb_kmeans)
        RGB_Means.append(rgb_mean)
        RGB_stds.append(rgb_std)
        HSV_Means.append(hsv_mean)
        HSV_stds.append(hsv_std)
        Lab_Means.append(lab_mean)
        Lab_stds.append(lab_std)
        Gray_Means.append(gray_mean)
        Gray_stds.append(gray_std)
        colorspaces.append(( rgb_mean, rgb_std, hsv_mean, hsv_std, lab_mean, lab_std,gray_mean,gray_std))

    # print("colorspaces", colorspaces)
    # dc.plotMultipleHistogram(0)
    # dc.plotMultipleHistogram(1)
    # dc.plotMultipleHistogram(2)

    #convert the list into numpy array#
    coords = np.array(coords)
    cn_Concentrations = np.array(cn_Concentrations)
    RGB_KMeans = np.array(RGB_KMeans)
    RGB_Means = np.array(RGB_Means)
    RGB_stds = np.array(RGB_stds)
    HSV_Means = np.array(HSV_Means)
    HSV_stds = np.array(HSV_stds)
    Lab_Means = np.array(Lab_Means)
    Lab_stds = np.array(Lab_stds)
    Gray_Means = np.array(Gray_Means)
    Gray_stds = np.array(Gray_stds)
    colorspaces= np.array(colorspaces)

    print("RGB KMEANS: ",RGB_KMeans)



    print("RGBKMEANS",RGB_KMeans)
    print("HSV MEANS: ",HSV_Means)
    ##convert the data type into a string##
    # ppm_values_str = ppm_values.astype(str).T
    cn_Concentrations_str = cn_Concentrations.astype(str).T
    RGB_KMeans_str = RGB_KMeans.astype(str).T
    RGB_Means_str = RGB_Means.astype(str).T
    RGB_stds_str = RGB_stds.astype(str).T
    HSV_Means_str = HSV_Means.astype(str).T
    HSV_stds_str = HSV_stds.astype(str).T          
    Lab_Means_str = Lab_Means.astype(str).T
    Lab_stds_str = Lab_stds.astype(str).T
    Gray_Means_str = Gray_Means.astype(str).T
    Gray_stds_str = Gray_stds.astype(str).T
    COLORSPACES_str = colorspaces.T

    ##stack the matrices vertically
    data = np.vstack((cn_Concentrations,coords,RGB_Means_str,RGB_stds_str,HSV_Means_str,HSV_stds_str,Lab_Means_str,Lab_stds_str,Gray_Means_str, Gray_stds_str))
    data2 = np.vstack((cn_Concentrations,coords, COLORSPACES_str))
    #save the data into a csv file.
    data, sorted_data = save_data(data, image_number,data_path, timestr, fnamee)
    pC.plotRGB(sorted_data, cn_Concentrations_str,data_path)
    pC.scatter_plotRGB(sorted_data, cn_Concentrations_str,data_path)
    pC.scatter_plotHSV(sorted_data, cn_Concentrations_str,data_path)
    pC.scatter_plotLAB(sorted_data, cn_Concentrations_str,data_path)
    pC.scatter_plotGRAY(sorted_data, cn_Concentrations_str,data_path)
    # print(colorspaces)
    print("lezgo")



if __name__ == '__main__':
    main()