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

# from openfolders_multiple import ROI_folder
# python color_extraction.py -id2 000 -rf2 ROI2_newsensor_45min
ap = argparse.ArgumentParser()
ap.add_argument("-id2","--images_id", required=False,
    help ="images id")
ap.add_argument("-rf2", "--ROI2_folder",required=False,
    help="filename of the ROI2 folder")

args = vars(ap.parse_args())
print(args)

images_id_no = args['images_id']
ROI2_folder= args['ROI2_folder']
fnamee =  ROI2_folder


CD = os.getcwd()
print(CD)

backCD =os.path.normpath(os.getcwd() + os.sep + os.pardir)
print(backCD)

DATA_DIRECTORY = os.path.join(backCD,"DATA")
print(DATA_DIRECTORY)

IMAGE_DIRECTORY = os.path.join(DATA_DIRECTORY,str(ROI2_folder))

IMAGE_DIRECTORY = os.path.join(IMAGE_DIRECTORY,str(images_id_no))

color_data_folder = os.path.join(DATA_DIRECTORY, "ColorData")
print(color_data_folder)
if not os.path.exists(color_data_folder):
    os.mkdir(color_data_folder)

color_data_folder = os.path.join(color_data_folder, ROI2_folder)
print(color_data_folder)

data_path = os.path.join(color_data_folder, images_id_no)
print(data_path)


histograms_path = os.path.join(data_path,"Histograms")

coord_no = "4"
if not os.path.exists(color_data_folder):
    os.mkdir(color_data_folder)

os.mkdir(data_path)
os.mkdir(histograms_path)


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
        # if coord == coord_no:
        images.append(image)
        ppm_values.append(cn_Concentration)
        combined.append((image, cn_Concentration))
        cn_Concentrations.append(cn_Concentration)
        coords.append(coord)

    return images, ppm_values, cn_Concentrations, coords

#function for saving the data
def save_data(data,image_number, timestr,fname):
    # print("data",data)
    
    data = data.T ##transpose the data array##
    # print("data",data)
    sorted_data = natsorted(data,key=itemgetter(0))##sort the data by their coordinates##
    # print("data",sorted_data)
    header = 'Cyanide Concentration,coordinate,R,G,B,R_std,G_std,B_std,H,S,V,H_std,S_std,V_std,L,a,b,L_std,a_std,b_std,Gray,Gray_std,RGB-KMEANS' #initialize the header for the csv file
    filename_Data =data_path+"\\"+image_number+fname + timestr+".csv" ##initialize the filename of the data
    filename_Sorted_Data =data_path+"\\"+image_number+fname + timestr+"x.csv"  ##initialize the filename of the sorted data
    data = np.array(data)## convert the data and sorted data into numpy arrays
    sorted_data = np.array(sorted_data)## convert the data and sorted data into numpy arrays
    # print(sorted_data)
    np.savetxt(filename_Data, data, delimiter=",",header= header,fmt='%s') #save the data array in a csv filetype with a filename of "data.csv" with the following header defined above
    np.savetxt(filename_Sorted_Data, sorted_data, delimiter=",",header= header,fmt='%s') #save the data array in a csv filetype with a filename of "data.csv" with the following header defined above

    return  data, sorted_data

#function for plotting the data
def plotRGB(sorted_data, ppm_values_str,data_path):

    fig1, (ax1, ax2) = plt.subplots(2, 1)

    #plot the RGB_Mean Intensity of the paper sensor that was taken
    ax1.plot(sorted_data[:,0].astype(float),sorted_data[:,2].astype(float),color='red', marker='x', linestyle='dashed')
    ax1.plot(sorted_data[:,0].astype(float),sorted_data[:,3].astype(float),color='green', marker='x', linestyle='dashed')
    ax1.plot(sorted_data[:,0].astype(float),sorted_data[:,4].astype(float),color='blue', marker='x', linestyle='dashed')
    ax1.set_ylabel('Mean Pixel Intensity')
    ax1.set_xlabel('Cyanide Concentration')
    ax1.set_title("Mean Pixel Intensity of Au-NP's in RGB Colorspace")
    ax1.legend()   
    labels = ppm_values_str

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars
    x =x+1

    rects1 = ax2.bar(x + width/2,sorted_data[:,2].astype(float),width, label='Red',color='r')
    rects2 = ax2.bar(x + 1.5*width, sorted_data[:,3].astype(float), width, label='Green',color='g')
    rects3 = ax2.bar(x + 2.5*width, sorted_data[:,4].astype(float), width, label='Blue',color='b')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax2.set_ylabel('Mean Pixel Intensity')
    ax2.set_title("Mean Pixel Intensity of Au-NP's in RGB Colorspace")
        
    ax2.set_xlabel('Cyanide Concentration(PPM)')
    ax2.legend()

    ax2.bar_label(rects1, padding=3)
    ax2.bar_label(rects2, padding=3)
    ax2.bar_label(rects3, padding=3)

    fig1.tight_layout()


    fig2, (ax21, ax22) = plt.subplots(2, 1)
    ax21.plot(sorted_data[:,0].astype(float),sorted_data[:,8].astype(float),color='red', marker='x', linestyle='dashed', label='Hue')
    ax21.plot(sorted_data[:,0].astype(float),sorted_data[:,9].astype(float),color='green', marker='x', linestyle='dashed', label='Saturation')
    ax21.plot(sorted_data[:,0].astype(float),sorted_data[:,10].astype(float),color='blue', marker='x', linestyle='dashed', label='Value')
    ax21.set_ylabel('Mean Pixel Intensity')
    ax21.set_xlabel('Cyanide Concentration(PPM)')
    ax21.set_title("Mean Pixel Intensity of Au-NP's in HSV Colorspace")
    ax21.legend()
    # plt.show()


    fig3, (ax31, ax32) = plt.subplots(2, 1)
    ax31.plot(sorted_data[:,0].astype(float),sorted_data[:,14].astype(float),color='red', marker='x', linestyle='dashed', label='L')
    ax31.plot(sorted_data[:,0].astype(float),sorted_data[:,15].astype(float),color='green', marker='x', linestyle='dashed', label='a')
    ax31.plot(sorted_data[:,0].astype(float),sorted_data[:,16].astype(float),color='blue', marker='x', linestyle='dashed', label='b')
    ax31.set_ylabel('Mean Pixel Intensity')
    ax31.set_xlabel('Cyanide Concentration(PPM)')
    ax31.set_title("Mean Pixel Intensity of Au-NP's in CIELAB Colorspace")
    ax31.legend()

    fig4, (ax41, ax42) = plt.subplots(2, 1)
    ax41.plot(sorted_data[:,0].astype(float),sorted_data[:,20].astype(float),color='red', marker='x', linestyle='dashed', label='Gray')
    ax41.set_ylabel('Mean Pixel Intensity')
    ax41.set_xlabel('CYANIDE Concentration (PPM)')
    ax41.set_title("Mean Pixel Intensity of Au-NP's in Gray Colorspace")

    ax41.legend()

    # plt.show()


    fig5, (ax51, ax52,ax53) = plt.subplots(3, 1)
    ax51.scatter(sorted_data[:,0].astype(float),sorted_data[:,2].astype(float),color='red', marker='x', label='Red')
    # ax51.set_ylabel('Mean Pixel Intensity')
    ax51.set_xlabel('CYANIDE Concentration (PPM)')
    ax51.set_title("RGB Colorspace")
    ax51.set_ylim(0, 270)
    ax51.set_xlim(0, 120)

    ax52.scatter(sorted_data[:,0].astype(float),sorted_data[:,3].astype(float),color='green', marker='x', label='Green')
    ax52.set_ylabel('Mean Pixel Intensity')
    ax52.set_xlabel('CYANIDE Concentration (PPM)')
  
    ax52.set_ylim(0, 270)
    ax52.set_xlim(0, 120)

    ax53.scatter(sorted_data[:,0].astype(float),sorted_data[:,4].astype(float),color='blue', marker='x', label='Blue')
    # ax53.set_ylabel('Mean Pixel Intensity')
    ax53.set_xlabel('CYANIDE Concentration (PPM)')
    
    ax53.set_ylim(0, 270)
    ax53.set_xlim(0, 120)
    ax51.legend()
    ax52.legend()
    ax53.legend()
    plt.savefig(data_path+'/RGBPLOT.png')
    # plt.show()
    print("Saved Plot")

    fig6, (ax61, ax62,ax63) = plt.subplots(3, 1)
    ax61.scatter(sorted_data[:,0].astype(float),sorted_data[:,8].astype(float),color='red', marker='x', label='Red')
    # ax61.set_ylabel('Mean Pixel Intensity')
    ax61.set_xlabel('CYANIDE Concentration (PPM)')
    ax61.set_title("HSV Colorspace")
    ax61.set_ylim(0, 270)
    ax61.set_xlim(-20, 120)

    ax62.scatter(sorted_data[:,0].astype(float),sorted_data[:,9].astype(float),color='green', marker='x', label='Green')
    ax62.set_ylabel('Mean Pixel Intensity')
    ax62.set_xlabel('CYANIDE Concentration (PPM)')
  
    ax62.set_ylim(0, 270)
    ax62.set_xlim(-20, 120)

    ax63.scatter(sorted_data[:,0].astype(float),sorted_data[:,10].astype(float),color='blue', marker='x', label='Blue')
    # ax63.set_ylabel('Mean Pixel Intensity')
    ax63.set_xlabel('CYANIDE Concentration (PPM)')
    
    ax63.set_ylim(0, 270)
    ax63.set_xlim(-20, 120)
    ax61.legend()
    ax62.legend()
    ax63.legend()
    plt.savefig(data_path+'/HSVPLOT.png')
    # plt.show()
    print("Saved Plot")

    
    fig7, (ax71, ax72,ax73) = plt.subplots(3, 1)
    ax71.scatter(sorted_data[:,0].astype(float),sorted_data[:,14].astype(float),color='red', marker='x', label='Red')
    # ax71.set_ylabel('Mean Pixel Intensity')
    ax71.set_xlabel('CYANIDE Concentration (PPM)')
    ax71.set_title("CIELAB Colorspace")
    ax71.set_ylim(0, 270)
    ax71.set_xlim(-20, 120)

    ax72.scatter(sorted_data[:,0].astype(float),sorted_data[:,15].astype(float),color='green', marker='x', label='Green')
    ax72.set_ylabel('Mean Pixel Intensity')
    ax72.set_xlabel('CYANIDE Concentration (PPM)')
  
    ax72.set_ylim(0, 270)
    ax72.set_xlim(-20, 120)

    ax73.scatter(sorted_data[:,0].astype(float),sorted_data[:,16].astype(float),color='blue', marker='x', label='Blue')
    # ax73.set_ylabel('Mean Pixel Intensity')
    ax73.set_xlabel('CYANIDE Concentration (PPM)')
    
    ax73.set_ylim(0, 270)
    ax73.set_xlim(-20, 120)
    ax71.legend()
    ax72.legend()
    ax73.legend()
    plt.savefig(data_path+'/CIELABPLOT.png')
    # plt.show()
    print("Saved Plot")

    ax81 = plt.figure()
    ax81_axes= ax81.add_subplot(111)
    ax81_axes.scatter(sorted_data[:,0].astype(float),sorted_data[:,14].astype(float),color='red', marker='x', label='Red')
    ax81_axes.set_ylabel('Mean Pixel Intensity')
    ax81_axes.set_xlabel('CYANIDE Concentration (PPM)')
    ax81_axes.set_title("Gray Colorspace")
    ax81_axes.set_ylim(0, 270)
    ax81_axes.set_xlim(-20, 120)

    plt.savefig(data_path+'/GRAYPLOT.png')
    plt.show()
    print("Saved Plot")


def scatter_plotRGB(sorted_data, ppm_values_str,data_path):
    fig_rgb = plt.figure()
    axes_rgb = fig_rgb.add_subplot(111)

    #plot the RGB_Mean Intensity of the paper sensor that was taken
    axes_rgb.scatter(sorted_data[:,0].astype(float),sorted_data[:,2].astype(float),color='red', marker='x', label = "Red")
    axes_rgb.scatter(sorted_data[:,0].astype(float),sorted_data[:,3].astype(float),color='green', marker='x', label = 'Green')
    axes_rgb.scatter(sorted_data[:,0].astype(float),sorted_data[:,4].astype(float),color='blue', marker='x', label= 'Blue')
    # for a,b in zip(sorted_data[:,0].astype(float).astype(int), sorted_data[:,2].astype(float).astype(int)): 
    #     plt.text(a+1, b+1, str(b))
    
    # for aa,bb in zip(sorted_data[:,0].astype(float).astype(int), sorted_data[:,3].astype(float).astype(int)): 
    #     plt.text(aa + 1, bb+1, str(bb))

    # for aaa,bbb in zip(sorted_data[:,0].astype(float).astype(int), sorted_data[:,4].astype(float).astype(int)): 
    #     plt.text(aaa+1, bbb+1, str(bbb))
    axes_rgb.set_ylabel('Mean Pixel Intensity')
    axes_rgb.set_xlabel('Cyanide Concentration (PPM)')
    axes_rgb.set_title("RGB Colorspace")
    axes_rgb.set_xlim(-20, 120)
    axes_rgb.set_ylim(0, 270)
    axes_rgb.legend()
    # plt.show()
    fig_rgb.savefig(data_path+'/RGBPLOT1.png')
    print("Saved Plot")

def scatter_plotHSV(sorted_data, ppm_values_str,data_path):
    fig_hsv = plt.figure()
    axes_hsv = fig_hsv.add_subplot(111)
    

    #plot the RGB_Mean Intensity of the paper sensor that was taken
    axes_hsv.scatter(sorted_data[:,0].astype(float),sorted_data[:,8].astype(float),color='red', marker='x', label = "Hue")
    axes_hsv.scatter(sorted_data[:,0].astype(float),sorted_data[:,9].astype(float),color='green', marker='x', label = 'Saturation')
    axes_hsv.scatter(sorted_data[:,0].astype(float),sorted_data[:,10].astype(float),color='blue', marker='x', label= 'Value')
    # for a,b in zip(sorted_data[:,0].astype(float).astype(int), sorted_data[:,8].astype(float).astype(int)): 
    #     plt.text(a+1, b+1, str(b))
    
    # for aa,bb in zip(sorted_data[:,0].astype(float).astype(int), sorted_data[:,9].astype(float).astype(int)): 
    #     plt.text(aa + 1, bb+1, str(bb))

    # for aaa,bbb in zip(sorted_data[:,0].astype(float).astype(int), sorted_data[:,10].astype(float).astype(int)): 
    #     plt.text(aaa+1, bbb+1, str(bbb))
    axes_hsv.set_ylabel('Mean Pixel Intensity')
    axes_hsv.set_xlabel('Cyanide Concentration (PPM)')
    axes_hsv.set_title("HSV Colorspace")
    axes_hsv.set_xlim(-20, 120)
    axes_hsv.set_ylim(0, 270)
    axes_hsv.legend()
    # plt.show()
    fig_hsv.savefig(data_path+'/HSVPLOT1.png')
    print("Saved Plot")

def scatter_plotLAB(sorted_data, ppm_values_str,data_path):
    
    fig_lab = plt.figure()
    axes_lab =fig_lab.add_subplot(111)

    #plot the RGB_Mean Intensity of the paper sensor that was taken
    axes_lab.scatter(sorted_data[:,0].astype(float),sorted_data[:,14].astype(float),color='red', marker='x', label = "L")
    axes_lab.scatter(sorted_data[:,0].astype(float),sorted_data[:,15].astype(float),color='green', marker='x', label = 'a*')
    axes_lab.scatter(sorted_data[:,0].astype(float),sorted_data[:,16].astype(float),color='blue', marker='x', label= 'b*')
    # for a,b in zip(sorted_data[:,0].astype(float).astype(int), sorted_data[:,14].astype(float).astype(int)): 
    #     plt.text(a+1, b+1, str(b))
    
    # for aa,bb in zip(sorted_data[:,0].astype(float).astype(int), sorted_data[:,15].astype(float).astype(int)): 
    #     plt.text(aa + 1, bb+1, str(bb))

    # for aaa,bbb in zip(sorted_data[:,0].astype(float).astype(int), sorted_data[:,16].astype(float).astype(int)): 
    #     plt.text(aaa+1, bbb+1, str(bbb))
    axes_lab.set_ylabel('Mean Pixel Intensity')
    axes_lab.set_xlabel('Cyanide Concentration (PPM)')
    axes_lab.set_title("CIELAB Colorspace")
    axes_lab.set_xlim(-20, 120)
    axes_lab.set_ylim(0, 270)
    axes_lab.legend()
    # plt.show()
    fig_lab.savefig(data_path+'/CIELABPLOT1.png')
    print("Saved Plot")

def scatter_plotGRAY(sorted_data, ppm_values_str,data_path):
    
    fig_gray = plt.figure()
    axes_gray = fig_gray.add_subplot(111)

    #plot the RGB_Mean Intensity of the paper sensor that was taken
    plt.scatter(sorted_data[:,0].astype(float),sorted_data[:,20].astype(float),color='red', marker='x', label = "Gray")
    # for a,b in zip(sorted_data[:,0].astype(float).astype(int), sorted_data[:,20].astype(float).astype(int)): 
    #     plt.text(a+1, b+1, str(b))
    
  
 
    axes_gray.set_ylabel('Mean Pixel Intensity')
    axes_gray.set_xlabel('Cyanide Concentration (PPM)')
    axes_gray.set_title("Gray Colorspace")
    axes_gray.set_xlim(-20, 120)
    axes_gray.set_ylim(0, 270)
    axes_gray.legend()
    # plt.show()
    fig_gray.savefig(data_path+'/GRAYPLOT1.png')
    print("Saved Plot")

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


    # _,_,_,image_number = IMAGE_DIRECTORY.split("\\")
    tail = os.path.split(IMAGE_DIRECTORY)
    image_number = tail[1]
    images , ppm_values, cn_Concentrations, coords = get_images_from_a_folder(IMAGE_DIRECTORY, coord_no)

    timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
    # print(timestr)



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
    data, sorted_data = save_data(data, image_number, timestr, fnamee)
    # data2, sorted_data2 = save_data(data2, image_number, timestr,'2')
    plotRGB(sorted_data, cn_Concentrations_str,data_path)
    scatter_plotRGB(sorted_data, cn_Concentrations_str,data_path)
    scatter_plotHSV(sorted_data, cn_Concentrations_str,data_path)
    scatter_plotLAB(sorted_data, cn_Concentrations_str,data_path)
    scatter_plotGRAY(sorted_data, cn_Concentrations_str,data_path)
    # print(colorspaces)
    print("lezgo")



if __name__ == '__main__':
    main()