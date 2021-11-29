# from Shapedetector import ShapeDetector
from operator import sub
# from shapedetector import ShapeDetector
import imutils
import cv2
import os
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import math
# Inked90ppmafter2min__LI
##C:\Users\CYANanoBot\Desktop\CYANanoBot- Analyzer>python paper_sensor_detection_program.py -ps paper_sensor_name -id2 000 -cif captured_images3 -nr 3 -si True
###folder where in you save the ROI's

###ARGUMENT PARSER
ap = argparse.ArgumentParser()

ap.add_argument("-ps","--paper_sensor", required=False,
    help ="paper sensor name")
ap.add_argument("-cif", "--captured_images_folder", required=False,
    help="folder name of the images gathered")
ap.add_argument("-si", "--show_image", required=False,
    help="show images")    
ap.add_argument("-nr","--number_rows", required=False, default= 3,type=int,
    help ="number of rows")    
args = vars(ap.parse_args())
print(args)
##########################################################################################################################


### save arguments to variables
FILENAME_DIR = args['paper_sensor']
ROI_DIR = 'ROI'
ROI2_DIR = 'ROI2'
NUM_ROWS = args['number_rows']
CAPTURED_IMAGES_DIR= args["captured_images_folder"] 
CAPTURED_IMAGES_SUB_DIR= args['paper_sensor']
show_image = args["show_image"]
DATA_DIR = "DATA"
##########################################################################################################################

### LOCATE and INITIALIZE DIRECTORIES 
CD = os.getcwd()
print(CD)
backCD =os.path.normpath(os.getcwd() + os.sep + os.pardir)
print(backCD)
DATA_PATH = os.path.join(backCD,DATA_DIR)
print(DATA_PATH)
FILENAME_PATH = os.path.join(DATA_PATH,FILENAME_DIR)
##########################################################################################################################


### lnitialize image path, roi path ###
ROI_PATH = os.path.join(FILENAME_PATH,str(ROI_DIR))
ROI2_PATH = os.path.join(FILENAME_PATH,str(ROI2_DIR))
IMAGES_PATH = os.path.join(DATA_PATH,str(CAPTURED_IMAGES_DIR))
print(IMAGES_PATH)
IMAGE_PATH = os.path.join(IMAGES_PATH,str(CAPTURED_IMAGES_SUB_DIR))
print(IMAGE_PATH)
##########################################################################################################################

### Create directories ###
if not os.path.exists(FILENAME_PATH):
    os.mkdir(FILENAME_PATH)

if not os.path.exists(ROI_PATH):
    os.mkdir(ROI_PATH)

if not os.path.exists(ROI2_PATH):
    os.mkdir(ROI2_PATH)
##########################################################################################################################




#initialize the list of images, and its filenames
image_no =  ["0000","000","01","02","03","04","05","06","07","08","09"]
##########################################################################################################################
print(image_no)
image_nos = image_no + list(map(str, range(88 + 1)))

print(image_nos)

#function for getting the image
def get_image(image_path):
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
##########################################################################################################################

### function for Paper Sensor Detector Algorithm
def get_circles(images,image_number, ppm_values, NUM_ROWS=3, show_image = False):

    ### Iterate the list of images
    for j in range(len(images)):
        print('image: ',j)
        img = images[j]
    ##########################################################################################################################    
    
        ### Added a mask so that the middle row will be highlighted and the other circles wont be detected
        if NUM_ROWS ==1:
            mask1 = np.zeros(img.shape[:2], dtype="uint8")
            cv2.rectangle(mask1, (0, 750), (2592, 1400), 255, -1)
            
            # cv2.rectangle(mask1, (0, 1200), (1944, 2592), 255, -1)

            img =  cv2.bitwise_and(img,img, mask=mask1)         #add the mask and the image

        ##########################################################################################################################    

        ### Circle Detection Algorithm ###

        cimg = img.copy()       # make a copy of the original image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert image to grayscale
        
        img = cv2.medianBlur(img, 5)    # apply a blur using the median filter
       
        circles = cv2.HoughCircles(    # finds the circles in the grayscale image using the Hough transform
            image=img,
            method=cv2.HOUGH_GRADIENT, 
            dp=1,
            minDist=500, 
            param1=40, 
            param2=23,
            minRadius=55, #the minimum radius of the paper sensor is 40 pixel
            maxRadius=78  #the maximum radius of the paper sensor is 60 pixel
        )
    
        ##########################################################################################################################    


        ### Sorting Circles Algorithm ###
        circles = np.round(circles[0, :]).astype("int")
        circles = sorted(circles, key=lambda v: [v[0], v[0]])
        print(circles)

        sorted_cols = []
        for k in range(0, len(circles), NUM_ROWS):
            col = circles[k:k+NUM_ROWS]
            sorted_cols.extend(sorted(col, key=lambda v: v[1]))

        circles = sorted_cols
        print(circles)

        circles = np.uint16(np.around(circles))
        ##########################################################################################################################    




        mask = np.zeros(shape=img.shape[0:2],dtype='uint8')        #initialize mask for the paper sensor

        #create mask for each paper sensor
        for co, i in enumerate(circles, start=1):
            cv2.circle(mask,(int(i[0]),int(i[1])),int(i[2]),(255,255,255),-1)         # draw the outer circle

        masked =  cv2.bitwise_and(cimg,cimg, mask=mask)         #add the mask and the image


        for co, i in enumerate(circles, start=1):          
            cv2.putText(masked,str(co),(int(i[0]+70),int(i[1])+70),cv2.FONT_HERSHEY_SIMPLEX, 1.5,(255,255,255),2)
            # cv2.circle(masked,(int(i[0]),int(i[1])),int(i[2]),(0,255,0),2)
            radius = int(math.ceil(i[2]))
            radius = 75
            origin_x = int(math.ceil(i[0]) - radius)
            origin_y = int(math.ceil(i[1]) - radius)
            cv2.rectangle(cimg,(origin_x-10,origin_y-10),(origin_x+2*radius+10,origin_y+2*radius+10),(200,0,0),2)
            roi=masked[origin_y:origin_y+2*radius,origin_x:origin_x+2*radius]   #crop the region of interest, roi is the circle, roi2 is a square within the paper sensor
            roi2=masked[origin_y+30:origin_y+2*radius-30,origin_x+30:origin_x+2*radius-30]

            #create ROI path
            # IMAGE_PATH = "C:\\Users\\CYANanoBot\\Desktop\\DATA\\"
            ROI_subfolder_path =   os.path.join( ROI_PATH,image_number,'')
            ROI2_subfolder_path =   os.path.join( ROI2_PATH,image_number,'')
            if not os.path.exists(ROI_subfolder_path):
                os.mkdir(ROI_subfolder_path)

            if not os.path.exists(ROI2_subfolder_path):
                os.mkdir(ROI2_subfolder_path)

            roi_path = ROI_subfolder_path + str(co) + ',' + str(ppm_values[j]) + '.png'
            roi2_path = ROI2_subfolder_path + str(co) + ',' + str(ppm_values[j]) + '.png'
            print('roi_path',roi_path)
            

            
            cv2.imwrite(roi_path, roi) ##save the cropped roi
            cv2.imwrite(roi2_path,roi2)##save the cropped roi
            cv2.waitKey(0)

          
       
        print("Number of circles detected:", co)     # print the number of circles detected


        ###Will display the image, mask, and masked image, and the 2 ROI if show_image == "True" (string not bool)
        if show_image == "True":
        # plt.imshow(masked)
            fig = plt.figure(figsize=(10, 7))
            rows= 1
            columns = 3
            #plt.show()

            fig.add_subplot(rows, columns, 1)
            
            # showing image
            plt.imshow(cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title("Captured Image")
            
            # Adds a subplot at the 2nd position
            fig.add_subplot(rows, columns, 2)
            
            # showing image
            plt.imshow(mask, cmap='gray')
            plt.axis('off')
            plt.title("Mask")
            
            # Adds a subplot at the 3rd position
            fig.add_subplot(rows, columns, 3)
            
            # showing image
            plt.imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title("Masked Image")
            plt.show()
            # Adds a subplot at the 4th position
            fig1 = plt.figure(figsize=(10, 7))
            
            fig1.add_subplot(1, 2, 1)
            
            # showing image
            plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title("Cropped Region of Interest")
            

            fig1.add_subplot(1, 2, 2)
            
            # showing image
            plt.imshow(cv2.cvtColor(roi2, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title("Cropped Region of Interest 2")
            plt.show()
            # #plt.show()
        ##########################################################################################################################    

def main():

    for image_no in image_nos:
        images = []
        ppm_values = []
        image_paths= []
        print(IMAGE_PATH)
        files = os.listdir(IMAGE_PATH)
        files = natsorted(files)

        ### This for loop will navigate you to the images from different folders(CN Concentration)
        for file in files:
            folder_path = os.path.join(IMAGE_PATH, file)  

            for subfolder in os.listdir(folder_path):
                subfolder_path = os.path.join(IMAGE_PATH,file, subfolder)

                for image in os.listdir(subfolder_path):
         
                    print('image_filename',image)
                    image_number, ppm_value,sec,seconds,second = image.split(',')
                    image_path = os.path.join(IMAGE_PATH,file, subfolder, image)

                    if image_number == image_no:
                        images.append(get_image(image_path))
                        image_paths.append(image_path)
                        ppm_values.append(ppm_value)
                        print(image_number)
                        print(image)
                        # print("diri2")
                # print(imagesss)
                
            # subfolders.append(subfolder)

        print('filenames',image_paths)
        print('ppm_values',ppm_values)
        print('image number: ',image_no)
        get_circles(images,image_no,ppm_values, NUM_ROWS, show_image)
        print(image_paths) 


if __name__ == '__main__':
    main()