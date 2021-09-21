from dominantColors import DominantColors
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from imutils import build_montages
from operator import itemgetter
import time




IMAGE_DIRECTORY = 'ROI/000' #Choose what certain time of images to be analyzed 'ROI/{id}' , id = [000,00,01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18,19,20]
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


#The filenames of the images of the ROI of paper sensor must be the PPM levels.
#the directory of the images of the paper sensor


def get_images_from_a_folder(path):
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
        print(file)
        image = get_image(os.path.join(IMAGE_DIRECTORY, file))

        images.append(image)
        ppm_value1, image_type = file.split('.')
        print(ppm_value1)
        coord, cn_Concentration = ppm_value1.split(',')
        # print(cn_Concentration)
        # lenStr = len(cn_Concentration)
        cn_Concentration = cn_Concentration[:-3]
        # print(cn_Concentration)


        cv2.putText(image, "{}".format(cn_Concentration), (5,5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,255,0), 1)
        
        ppm_values.append(cn_Concentration)
        combined.append((image, cn_Concentration))
        cn_Concentrations.append(cn_Concentration)
        coords.append(coord)
    combined = [r[0] for r in combined[:8]]
    ppm_values = np.array(ppm_values) #convert the list to numpy array
    # print(ppm_values)
    # print(combined)
    mostColorMontage = build_montages(combined, (128,128), (2,4))
#     cv2.imshow("Most Colorful",mostColorMontage[0])
#     cv2.waitKey(0)
    return images, ppm_values, cn_Concentrations, coords



def save_data(data):
    print("data",data)
    data = data.T ##transpose the data array##
    print("data",data)
    sorted_data = natsorted(data,key=itemgetter(0))##sort the data by their coordinates##
    print("data",sorted_data)
    header = 'Cyanide Concentration,R,G,B,R_std,G_std,B_std,H,S,V,H_std,S_std,V_std,L,a,b,L_std,a_std,b_std,Gray,Gray_std,RGB-KMEANS' #initialize the header for the csv file
    filename_Data ="ColorData/"+image_number+ "_ColorData_" + timestr+".csv" ##initialize the filename of the data
    filename_Sorted_Data ="ColorData/"+image_number+ "_ColorSortedData_" + timestr+".csv"  ##initialize the filename of the sorted data
    data = np.array(data)## convert the data and sorted data into numpy arrays
    sorted_data = np.array(sorted_data)## convert the data and sorted data into numpy arrays
    print(sorted_data)
    np.savetxt(filename_Data, data, delimiter=",",header= header,fmt='%s') #save the data array in a csv filetype with a filename of "data.csv" with the following header defined above
    np.savetxt(filename_Sorted_Data, sorted_data, delimiter=",",header= header,fmt='%s') #save the data array in a csv filetype with a filename of "data.csv" with the following header defined above

    return  data, sorted_data



#for KMeans Algorithm
clusters = 1 #number of clusters of colors. Value is at the range of (2-5)
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
images , ppm_values, cn_Concentrations, coords = get_images_from_a_folder(IMAGE_DIRECTORY)

timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
print(timestr)



print(len(images))

for i in range(len(images)):

    dc = DominantColors(images[i],images,clusters) #initialize the DominantColors class
    dc.saveHistogram("Histograms/{}Histogram".format(i), False)

    colors = dc.dominantColors()  #call the dominantColors function to get the dominant colors of the image using KMeans Algorithm
    print("Dominant Colors: ",colors)
    print("Dominant Colors sorted: ",colors)
    hsv,lab,gray = dc.cvtColorSpace()
    # dc.plotHistogram()
    rgb_mean,rgb_std,hsv_mean,hsv_std,lab_mean,lab_std, gray_mean, gray_std = dc.getMeanIntensity()  #call the getMeanIntensity function to get the average RGB pixel intensity and its standard deviation of the paper sensor

    #append the RGB, HSV,Lab, Gray Values into a list
    RGB_KMeans.append(colors)
    RGB_Means.append(rgb_mean)
    RGB_stds.append(rgb_std)
    HSV_Means.append(hsv_mean)
    HSV_stds.append(hsv_std)
    Lab_Means.append(lab_mean)
    Lab_stds.append(lab_std)
    Gray_Means.append(gray_mean)
    Gray_stds.append(gray_std)
    colorspaces.append((colors, rgb_mean, rgb_std, hsv_mean, hsv_std, lab_mean, lab_std,gray_mean,gray_std))

# dc.plotMultipleHistogram(0)
# dc.plotMultipleHistogram(1)
# dc.plotMultipleHistogram(2)

#convert the list into numpy array#
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
print('Colorspaces',colorspaces)
print("RGB KMEANS: ",RGB_KMeans)
print("RGB MEANS: ",RGB_Means)
# print("RGB STDS: ",RGB_stds)
# print("HSV MEANS: ",HSV_Means)
# print("HSV STDS: ",HSV_stds)
# print("Lab MEANS: ",Lab_Means)
# print("Lab STDS: ",Lab_stds)
# print("Gray MEANS: ",Gray_Means)
# print("Gray STDS: ",Gray_stds)


# HSV_Means[:,0] = HSV_Means[:,0]/180*360
# HSV_Means[:,1:] = np.round(HSV_Means[:,1:]/255,8)
# HSV_stds[:,0] = HSV_stds[:,0]/180*360
# HSV_stds[:,1:] = np.round(HSV_stds[:,1:]/255,8)
RGB_KMeans = np.squeeze(RGB_KMeans[:][:,0:2])
red = RGB_Means[:,0]
green = RGB_Means[:,1]
blue = RGB_Means[:,2]
print("RGBKMEANS",RGB_KMeans)
print("HSV MEANS: ",HSV_Means)
##convert the data type into a string##
ppm_values_str = ppm_values.astype(str).T
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
# print("cn_Concentrations",cn_Concentrations)
# print("ppm",ppm_values_str)
# print("means",RGB_Means_str)
# print("stds",RGB_stds_str)

##stack the matrices vertically
data = np.vstack((cn_Concentrations,RGB_Means_str,RGB_stds_str,HSV_Means_str,HSV_stds_str,Lab_Means_str,Lab_stds_str,Gray_Means_str, Gray_stds_str,RGB_KMeans_str))
data, sorted_data = save_data(data)

# print("weee")
# print(sorted_data[:,0])
# print(sorted_data[:,1])
# print(sorted_data[:,2])
# print(sorted_data[:,3])
# print("weee")


# fig1, (ax1, ax2) = plt.subplots(2, 1)

# #plot the RGB_Mean Intensity of the paper sensor that was taken
# ax1.plot(ppm_values,RGB_Means[:,0],color='red', marker='o', linestyle='dashed')
# ax1.plot(ppm_values,RGB_Means[:,1],color='green', marker='o', linestyle='dashed')
# ax1.plot(ppm_values,RGB_Means[:,2],color='blue', marker='o', linestyle='dashed')
# ax1.set_ylabel('Mean Pixel Intensity')
# ax1.set_xlabel('Cyanide Concentration')
# ax1.set_title("Mean Pixel Intensity of Au-NP's in RGB Colorspace")
# ax1.legend()





      
# labels = ppm_values_str
# # men_means = [20, 34, 30, 35, 27]
# # women_means = [25, 32, 34, 20, 25]

# x = np.arange(len(labels))  # the label locations
# width = 0.2  # the width of the bars
# x =x+1
# # print('x',x)
# # print(labels)
# # fig, ax1 = plt.subplots()
# rects1 = ax2.bar(x + width/2,RGB_Means[:,0],width, label='Red',color='r')
# rects2 = ax2.bar(x + 1.5*width, RGB_Means[:,1], width, label='Green',color='g')
# rects3 = ax2.bar(x + 2.5*width, RGB_Means[:,2], width, label='Blue',color='b')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax2.set_ylabel('Mean Pixel Intensity')
# ax2.set_title("Mean Pixel Intensity of Au-NP's in RGB Colorspace")
      
# ax2.set_xlabel('Cyanide Concentration(PPM)')
# ax2.legend()

# ax2.bar_label(rects1, padding=3)
# ax2.bar_label(rects2, padding=3)
# ax2.bar_label(rects3, padding=3)

# fig1.tight_layout()


# #
# # plt.show()


# fig2, (ax21, ax22) = plt.subplots(2, 1)
# ax21.plot(ppm_values,HSV_Means[:,0],color='red', marker='o', linestyle='dashed', label='Hue')
# ax21.plot(ppm_values,HSV_Means[:,1],color='green', marker='o', linestyle='dashed', label='Saturation')
# ax21.plot(ppm_values,HSV_Means[:,2],color='blue', marker='o', linestyle='dashed', label='Value')
# ax21.set_ylabel('Mean Pixel Intensity')
# ax21.set_xlabel('Cyanide Concentration(PPM)')
# ax21.set_title("Mean Pixel Intensity of Au-NP's in HSV Colorspace")
# ax21.legend()
# # plt.show()


# fig3, (ax31, ax32) = plt.subplots(2, 1)
# ax31.plot(ppm_values,Lab_Means[:,0],color='red', marker='o', linestyle='dashed', label='L')
# ax31.plot(ppm_values,Lab_Means[:,1],color='green', marker='o', linestyle='dashed', label='a')
# ax31.plot(ppm_values,Lab_Means[:,2],color='blue', marker='o', linestyle='dashed', label='b')
# ax31.set_ylabel('Mean Pixel Intensity')
# ax31.set_xlabel('Cyanide Concentration(PPM)')
# ax31.set_title("Mean Pixel Intensity of Au-NP's in LAB Colorspace")
# ax31.legend()
# # plt.show()

# # scatter_color = RGB_Means/255
# # print(scatter_color)
# # area = 500  # 0 to 15 point radii

# # plt.scatter(ppm_values, RGB_Means[:,0], s=area, c=scatter_color, alpha=0.5)

# # plt.show()


# # scatter_hsv = HSV_Means[:,0]
# # print(scatter_hsv)
# # area = 500  # 0 to 15 point radii

# # plt.scatter(ppm_values, HSV_Means[:,0], s=area, c=scatter_hsv, alpha=0.5)
# # plt.show()
# fig4, (ax41, ax42) = plt.subplots(2, 1)
# ax41.plot(ppm_values,Gray_Means[:,0],color='red', marker='o', linestyle='dashed', label='Gray')
# ax41.set_ylabel('Mean Pixel Intensity')
# ax41.set_xlabel('CYANIDE Concentration (PPM)')
# ax41.set_title("Mean Pixel Intensity of Au-NP's in Gray Colorspace")

# ax41.legend()

# fig, ax5 = plt.subplots(3, figsize=(10, 6))
# ax5[0].scatter(sorted_data[:,0].astype(float), sorted_data[:,1].astype(float))
# ax5[0].set_xlabel("Cyanide Concentration")
# ax5[0].set_ylabel("Mean Pixel Intentisity")

# ax5[1].scatter(sorted_data[:,0].astype(float), sorted_data[:,2].astype(float))
# ax5[1].set_xlabel("Cyanide Concentration")
# ax5[1].set_ylabel("Mean Pixel Intentisity")

# ax5[2].scatter(sorted_data[:,0].astype(float), sorted_data[:,3].astype(float))
# ax5[2].set_xlabel("Cyanide Concentration")
# ax5[2].set_ylabel("Mean Pixel Intentisity")


# fig, ax6 = plt.subplots(3, figsize=(10, 6))
# ax6[0].scatter(data[:,0].astype(float), data[:,1].astype(float))
# ax6[0].set_xlabel("Cyanide Concentration")
# ax6[0].set_ylabel("Mean Pixel Intentisity")

# ax6[1].scatter(data[:,0].astype(float),  data[:,2].astype(float))
# ax6[1].set_xlabel("Cyanide Concentration")
# ax6[1].set_ylabel("Mean Pixel Intentisity")

# ax6[2].scatter( data[:,0].astype(float),  data[:,3].astype(float))
# ax6[2].set_xlabel("Cyanide Concentration")
# ax6[2].set_ylabel("Mean Pixel Intentisity")
# plt.show()



fig1, (ax1, ax2) = plt.subplots(2, 1)

#plot the RGB_Mean Intensity of the paper sensor that was taken
ax1.plot(sorted_data[:,0].astype(float),sorted_data[:,1].astype(float),color='red', marker='o', linestyle='dashed')
ax1.plot(sorted_data[:,0].astype(float),sorted_data[:,2].astype(float),color='green', marker='o', linestyle='dashed')
ax1.plot(sorted_data[:,0].astype(float),sorted_data[:,3].astype(float),color='blue', marker='o', linestyle='dashed')
ax1.set_ylabel('Mean Pixel Intensity')
ax1.set_xlabel('Cyanide Concentration')
ax1.set_title("Mean Pixel Intensity of Au-NP's in RGB Colorspace")
ax1.legend()





      
labels = ppm_values_str
# men_means = [20, 34, 30, 35, 27]
# women_means = [25, 32, 34, 20, 25]

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars
x =x+1
print('x',x)
print(labels)
# fig, ax1 = plt.subplots()
rects1 = ax2.bar(x + width/2,RGB_Means[:,0],width, label='Red',color='r')
rects2 = ax2.bar(x + 1.5*width, RGB_Means[:,1], width, label='Green',color='g')
rects3 = ax2.bar(x + 2.5*width, RGB_Means[:,2], width, label='Blue',color='b')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax2.set_ylabel('Mean Pixel Intensity')
ax2.set_title("Mean Pixel Intensity of Au-NP's in RGB Colorspace")
      
ax2.set_xlabel('Cyanide Concentration(PPM)')
ax2.legend()

ax2.bar_label(rects1, padding=3)
ax2.bar_label(rects2, padding=3)
ax2.bar_label(rects3, padding=3)

fig1.tight_layout()


#
# plt.show()


fig2, (ax21, ax22) = plt.subplots(2, 1)
ax21.plot(sorted_data[:,0].astype(float),sorted_data[:,7].astype(float),color='red', marker='o', linestyle='dashed', label='Hue')
ax21.plot(sorted_data[:,0].astype(float),sorted_data[:,8].astype(float),color='green', marker='o', linestyle='dashed', label='Saturation')
ax21.plot(sorted_data[:,0].astype(float),sorted_data[:,9].astype(float),color='blue', marker='o', linestyle='dashed', label='Value')
ax21.set_ylabel('Mean Pixel Intensity')
ax21.set_xlabel('Cyanide Concentration(PPM)')
ax21.set_title("Mean Pixel Intensity of Au-NP's in HSV Colorspace")
ax21.legend()
# plt.show()


fig3, (ax31, ax32) = plt.subplots(2, 1)
ax31.plot(sorted_data[:,0].astype(float),sorted_data[:,13].astype(float),color='red', marker='o', linestyle='dashed', label='L')
ax31.plot(sorted_data[:,0].astype(float),sorted_data[:,14].astype(float),color='green', marker='o', linestyle='dashed', label='a')
ax31.plot(sorted_data[:,0].astype(float),sorted_data[:,15].astype(float),color='blue', marker='o', linestyle='dashed', label='b')
ax31.set_ylabel('Mean Pixel Intensity')
ax31.set_xlabel('Cyanide Concentration(PPM)')
ax31.set_title("Mean Pixel Intensity of Au-NP's in LAB Colorspace")
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
fig4, (ax41, ax42) = plt.subplots(2, 1)
ax41.plot(sorted_data[:,0].astype(float),sorted_data[:,19].astype(float),color='red', marker='o', linestyle='dashed', label='Gray')
ax41.set_ylabel('Mean Pixel Intensity')
ax41.set_xlabel('CYANIDE Concentration (PPM)')
ax41.set_title("Mean Pixel Intensity of Au-NP's in Gray Colorspace")

ax41.legend()
plt.show()
