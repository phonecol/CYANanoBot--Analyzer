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




IMAGE_DIRECTORY = "C:\\Users\\CYANanoBot\\Desktop\\DATA\\ROI_2/20" #Choose what certain time of images to be analyzed 'ROI/{id}' , id = [000,00,01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18,19,20]
coord_nos = ["4","5","6","8"]
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
    files = os.listdir(path)
    files = natsorted(files)
    print(files)

    #a loop for getting the images in the folder and append them into a list
    for file in files:
        image = get_image(os.path.join(IMAGE_DIRECTORY, file))
        # ppm_value1, image_type = file.split('.')#split the filename to remove the image type
        ppm_value1 = file[:-4]
        coord, cn_Concentration = ppm_value1.split(',')#split the coordinate of the paper sensor in the spotplate and the cyanide concentration
        cn_Concentration = cn_Concentration[:-3]

        #append the images, cyanide concentrations, and coordinates
        if coord == coord_no:
            images.append(image)
            ppm_values.append(cn_Concentration)
            combined.append((image, cn_Concentration))
            cn_Concentrations.append(cn_Concentration)
            coords.append(coord)
            # combined = [r[0] for r in combined[:8]]
            # ppm_values = np.array(ppm_values) #convert the list to numpy array

            # mostColorMontage = build_montages(combined, (128,128), (2,4))
#     cv2.imshow("Most Colorful",mostColorMontage[0])
#     cv2.waitKey(0)
    return images, ppm_values, cn_Concentrations, coords


#function for saving the data
def save_data(data,image_number, timestr,fname):
    # print("data",data)
    
    data = data.T ##transpose the data array##
    # print("data",data)
    sorted_data = natsorted(data,key=itemgetter(0))##sort the data by their coordinates##
    # print("data",sorted_data)
    header = 'Cyanide Concentration,coordinate,R,G,B,R_std,G_std,B_std,H,S,V,H_std,S_std,V_std,L,a,b,L_std,a_std,b_std,Gray,Gray_std,RGB-KMEANS' #initialize the header for the csv file
    filename_Data ="ColorData/"+image_number+ "_ColorData_"+fname + timestr+".csv" ##initialize the filename of the data
    filename_Sorted_Data ="ColorData/"+image_number+ "_ColorSortedData_"+fname + timestr+".csv"  ##initialize the filename of the sorted data
    data = np.array(data)## convert the data and sorted data into numpy arrays
    sorted_data = np.array(sorted_data)## convert the data and sorted data into numpy arrays
    # print(sorted_data)
    np.savetxt(filename_Data, data, delimiter=",",header= header,fmt='%s') #save the data array in a csv filetype with a filename of "data.csv" with the following header defined above
    np.savetxt(filename_Sorted_Data, sorted_data, delimiter=",",header= header,fmt='%s') #save the data array in a csv filetype with a filename of "data.csv" with the following header defined above

    return  data, sorted_data


#function for plotting the data
def plotRGB(sorted_data, ppm_values_str):
    

    #plot the RGB_Mean Intensity of the paper sensor that was taken
    plt.plot(sorted_data[:,0].astype(float),sorted_data[:,2].astype(float),color='red', marker='o', linestyle='dashed')
    plt.plot(sorted_data[:,0].astype(float),sorted_data[:,3].astype(float),color='green', marker='o', linestyle='dashed')
    plt.plot(sorted_data[:,0].astype(float),sorted_data[:,4].astype(float),color='blue', marker='o', linestyle='dashed')
    plt.ylabel('Mean Pixel Intensity')
    plt.xlabel('Cyanide Concentration (PPM)')
    plt.title("RGB Colorspace")
    plt.legend()
    plt.show()

  
    # plt.plot(sorted_data[:,0].astype(float),sorted_data[:,8].astype(float),color='red', marker='o', linestyle='dashed', label='Hue')
    # plt.plot(sorted_data[:,0].astype(float),sorted_data[:,9].astype(float),color='green', marker='o', linestyle='dashed', label='Saturation')
    # plt.plot(sorted_data[:,0].astype(float),sorted_data[:,10].astype(float),color='blue', marker='o', linestyle='dashed', label='Value')
    # plt.ylabel('Mean Pixel Intensity')
    # plt.xlabel('Cyanide Concentration(PPM)')
    # plt.title("HSV Colorspace")
    # plt.legend()
    # plt.show()


    # plt.plot(sorted_data[:,0].astype(float),sorted_data[:,14].astype(float),color='red', marker='o', linestyle='dashed', label='L')
    # plt.plot(sorted_data[:,0].astype(float),sorted_data[:,15].astype(float),color='green', marker='o', linestyle='dashed', label='a')
    # plt.plot(sorted_data[:,0].astype(float),sorted_data[:,16].astype(float),color='blue', marker='o', linestyle='dashed', label='b')
    # plt.ylabel('Mean Pixel Intensity')
    # plt.xlabel('Cyanide Concentration(PPM)')
    # plt.title("LAB Colorspace")
    # plt.legend()
    # plt.show()


    # plt.plot(sorted_data[:,0].astype(float),sorted_data[:,20].astype(float),color='red', marker='o', linestyle='dashed', label='Gray')
    # plt.ylabel('Mean Pixel Intensity')
    # plt.xlabel('CYANIDE Concentration (PPM)')
    # plt.title("Gray Colorspace")
    # plt.legend()

    # plt.show()


def plotCOLORSPACES(sorted_data, ppm_values_str):
    fig1, ((ax1, ax21), (ax31, ax41)) = plt.subplots(2, 2)

    #plot the RGB_Mean Intensity of the paper sensor that was taken
    ax1.plot(sorted_data[:,0].astype(float),sorted_data[:,2].astype(float),color='red', marker='o', linestyle='dashed')
    ax1.plot(sorted_data[:,0].astype(float),sorted_data[:,3].astype(float),color='green', marker='o', linestyle='dashed')
    ax1.plot(sorted_data[:,0].astype(float),sorted_data[:,4].astype(float),color='blue', marker='o', linestyle='dashed')
    ax1.set_ylabel('Mean Pixel Intensity')
    ax1.set_xlabel('Cyanide Concentration (PPM)')
    ax1.set_title("RGB Colorspace")
    ax1.legend()





        
    labels = ppm_values_str


   


  
    ax21.plot(sorted_data[:,0].astype(float),sorted_data[:,8].astype(float),color='red', marker='o', linestyle='dashed', label='Hue')
    ax21.plot(sorted_data[:,0].astype(float),sorted_data[:,9].astype(float),color='green', marker='o', linestyle='dashed', label='Saturation')
    ax21.plot(sorted_data[:,0].astype(float),sorted_data[:,10].astype(float),color='blue', marker='o', linestyle='dashed', label='Value')
    ax21.set_ylabel('Mean Pixel Intensity')
    ax21.set_xlabel('Cyanide Concentration(PPM)')
    ax21.set_title("HSV Colorspace")
    ax21.legend()
    # plt.show()


 
    ax31.plot(sorted_data[:,0].astype(float),sorted_data[:,14].astype(float),color='red', marker='o', linestyle='dashed', label='L')
    ax31.plot(sorted_data[:,0].astype(float),sorted_data[:,15].astype(float),color='green', marker='o', linestyle='dashed', label='a')
    ax31.plot(sorted_data[:,0].astype(float),sorted_data[:,16].astype(float),color='blue', marker='o', linestyle='dashed', label='b')
    ax31.set_ylabel('Mean Pixel Intensity')
    ax31.set_xlabel('Cyanide Concentration(PPM)')
    ax31.set_title("LAB Colorspace")
    ax31.legend()
    # plt.show()

    # scatter_color = RGB_Means/255
    # print(scatter_color)
    # area = 500  # 0 to 15 point radii

    # plt.scatter(ppm_values, RGB_Means[:,0], s=area, c=scatter_color, alpha=0.5)

    # plt.show()


    # scatter_hsv = HSV_Means[:,0]
    # print(scatter_hsv)
    # area = 500  # 0 to 15 point radii

    # plt.scatter(ppm_values, HSV_Means[:,0], s=area, c=scatter_hsv, alpha=0.5)
    # plt.show()

    ax41.plot(sorted_data[:,0].astype(float),sorted_data[:,20].astype(float),color='red', marker='o', linestyle='dashed', label='Gray')
    ax41.set_ylabel('Mean Pixel Intensity')
    ax41.set_xlabel('CYANIDE Concentration (PPM)')
    ax41.set_title("Gray Colorspace")

    ax41.legend()

    plt.show()



#for KMeans Algorithm
for coord_no in coord_nos:
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


    _,image_number = IMAGE_DIRECTORY.split('/')
    images , ppm_values, cn_Concentrations, coords = get_images_from_a_folder(IMAGE_DIRECTORY, coord_no)

    timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
    # print(timestr)



    # print(len(images))

    for i in range(len(images)):

        dc = DominantColors(images[i],images,clusters) #initialize the DominantColors class
        dc.saveHistogram("Histograms/{}Histogram".format(i), False) #set to false para dili i show ang histogram plot

        rgb_kmeans = dc.dominantColors()  #call the dominantColors function to get the dominant colors of the image using KMeans Algorithm
        rgb_kmeans = rgb_kmeans.flatten()
        print("Dominant Colors: ",rgb_kmeans)
        print("Dominant Colors sorted: ",rgb_kmeans)
        hsv,lab,gray = dc.cvtColorSpace()
        # dc.plotHistogram()s
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
        colorspaces.append(( rgb_mean, rgb_std, hsv_mean, hsv_std, lab_mean, lab_std,gray_mean,gray_std,rgb_kmeans))

    print("colorspaces", colorspaces)
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
    data = np.vstack((cn_Concentrations,coords,RGB_Means_str,RGB_stds_str,HSV_Means_str,HSV_stds_str,Lab_Means_str,Lab_stds_str,Gray_Means_str, Gray_stds_str,RGB_KMeans_str))
    data2 = np.vstack((cn_Concentrations,coords, COLORSPACES_str))
    #save the data into a csv file.
    data, sorted_data = save_data(data, image_number, timestr, "1")
    # data2, sorted_data2 = save_data(data2, image_number, timestr,'2')
    plotRGB(sorted_data, cn_Concentrations_str)

    print(colorspaces)
    print("lezgo")