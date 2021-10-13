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

###folder where in you save the ROI's
ap = argparse.ArgumentParser()
ap.add_argument("-id2","--images_id", required=False,
    help ="images id")
ap.add_argument("-rf1","--ROI_folder", required=False,
    help ="filename of the ROI folder")
ap.add_argument("-rf2", "--ROI2_folder",required=False,
    help="filename of the ROI2 folder")
ap.add_argument("-cif", "--captured_images_folder", required=False,
    help="folder name of the images gathered")
ap.add_argument("-cisf", "----captured_images_subfolder", required=False,
    help="subfolder name of the images gathered")
args = vars(ap.parse_args())
print(args)



# ROI_folder = "ROI_45min"
# ROI2_folder = "ROI2_45min"

ROI_folder = args['ROI_folder']
ROI2_folder = args['ROI2_folder']
#load image
IMAGE_PATH = "C:\\Users\\CYANanoBot\\Desktop\\DATA\\"


# IMAGE_DIRECTORY = os.path.join('captured_images1', 'optimizationA')
# CAPTURED_IMAGES_FOLDER= "captured_images3" 
# CAPTURED_IMAGES_SUBFOLDER= "data_gathering_45min" 

CAPTURED_IMAGES_FOLDER= args["captured_images_folder"] 
CAPTURED_IMAGES_SUBFOLDER= args["captured_images_subfolder"] 


IMAGE_DIRECTORY = IMAGE_PATH+str(CAPTURED_IMAGES_FOLDER)+ "/"+ str(CAPTURED_IMAGES_SUBFOLDER)

#  id = [000,00,01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18,19,20,...29]

#initialize the list of images, and its filenames
image_nos =  ["000","01","02","03","04","05","06","07","08","09","10"]



#function for getting the image
def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    return image


def get_circles(images,image_number, ppm_values):

    for j in range(len(images)):
        print('image: ',j)
        img = images[j]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # make a copy of the original image
        cimg = img.copy()
        # convert image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # apply a blur using the median filter
        img = cv2.medianBlur(img, 5)
        # finds the circles in the grayscale image using the Hough transform
        circles = cv2.HoughCircles(
            image=img,
            method=cv2.HOUGH_GRADIENT, 
            dp=1,
            minDist=500, 
            param1=40, 
            param2=23,
            minRadius=65, #the minimum radius of the paper sensor is 40 pixel
            maxRadius=78  #the maximum radius of the paper sensor is 60 pixel
        )
        # print(circles)
        # print("asd")
        # print(circles[0,:,0])
        # print("asd")
        NUM_ROWS = 3
        #sort circles
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

        #initialize mask for the paper sensor
        mask = np.zeros(shape=img.shape[0:2],dtype='uint8')

        #create mask for each paper sensor
        for co, i in enumerate(circles, start=1):
            # draw the outer circle
            cv2.circle(mask,(int(i[0]),int(i[1])),int(i[2]),(255,255,255),-1)

        # cv2.imshow('mask',mask)
        # cv2.waitKey(0)
        
        # plt.imshow(mask,"bgr")
        # plt.show()
       
        #add the mask and the image
        masked =  cv2.bitwise_and(cimg,cimg, mask=mask)

        # cv2.imshow('masked',masked)
        # cv2.waitKey(0)

        for co, i in enumerate(circles, start=1):
            # draw the outer circle
            print(co)
            # print(i[0])
            # print(i[1])
            print(i[2])
            
            cv2.putText(masked,str(co),(int(i[0]+70),int(i[1])+70),cv2.FONT_HERSHEY_SIMPLEX, 1.5,(255,255,255),2)
            # cv2.circle(masked,(int(i[0]),int(i[1])),int(i[2]),(0,255,0),2)
            
            radius = int(math.ceil(i[2]))
            radius = 75
            origin_x = int(math.ceil(i[0]) - radius)
            origin_y = int(math.ceil(i[1]) - radius)
            

            cv2.rectangle(cimg,(origin_x-10,origin_y-10),(origin_x+2*radius+10,origin_y+2*radius+10),(200,0,0),2)

            #cropped the region of interest, roi is the circle, roi2 is a square within the paper sensor
            roi=masked[origin_y:origin_y+2*radius,origin_x:origin_x+2*radius]
            roi2=masked[origin_y+30:origin_y+2*radius-30,origin_x+30:origin_x+2*radius-30]

            #roi path
            IMAGE_PATH = "C:\\Users\\CYANanoBot\\Desktop\\DATA\\"
            roi_path = IMAGE_PATH+ ROI_folder+"\\" +image_number+'\\'+str(co)+','+str(ppm_values[j]) + '.jpg'
            roi2_path = IMAGE_PATH+ROI2_folder+"\\" +image_number+'\\'+str(co)+','+str(ppm_values[j]) + '.jpg'
            print('roi_path',roi_path)
            
            cv2.imwrite(roi_path, roi)
            cv2.imwrite(roi2_path,roi2)

            # cv2.imwrite("ROI2\\"+image_number+'\\'+str(co)+','+str(ppm_values[j])+ '.jpg', roi2)
            cv2.waitKey(0)

            # roi=cimg[y:y+h,x:x+w]
        # print the number of circles detected
        print("Number of circles detected:", co)

        # plt.imshow(masked)
        # plt.show()
        # #plt.show()

def main():

    for image_no in image_nos:
        images = []
        ppm_values = []
        subfolders = []
        image_folders = []
        imagesss = []
        imagessss= []
        imagesssss = []
        image_paths= []
        files = os.listdir(IMAGE_DIRECTORY)
        files = natsorted(files)


        for file in files:
            # print(file)
            folder_path = os.path.join(IMAGE_DIRECTORY, file)  
            
            for subfolder in os.listdir(folder_path):
                # print(subfolder)
                subfolder_path = os.path.join(IMAGE_DIRECTORY,file, subfolder)
                # print(subfolder_path)
                for image in os.listdir(subfolder_path): 
                
                    print('image_filename',image)
                    image_number, ppm_value,sec,seconds,second = image.split(',')
                    # print(image_number)
                    image_path = os.path.join(IMAGE_DIRECTORY,file, subfolder, image)
                    # print(image_path)
                    if image_number == image_no:
                    
                        images.append(get_image(image_path))
                        image_paths.append(image_path)
                        ppm_values.append(ppm_value)
                        print(image_number)
                        print(image)
                        print("diri2")
                # print(imagesss)
                
            # subfolders.append(subfolder)

        print('filenames',image_paths)
        print('ppm_values',ppm_values)
        print("lezgoo")
        print('image number: ',image_no)
        get_circles(images,image_no,ppm_values)
        # print(imagesss)
        print("lezgoo")
        print(image_paths) 

        print("lezgoooooooooo")




if __name__ == '__main__':
    main()