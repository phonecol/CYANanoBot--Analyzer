from time import sleep, time

from picamera import PiCamera
import numpy as np
import cv2
import datetime as dt
import os
import argparse
#$ python image_acquisition_program.py -cn 100ppm -ps paper_sensor_name -i 30 -cif captured_images3 -e 100 -in 30 -r H 
ap = argparse.ArgumentParser()
# 
ap.add_argument("-cn","--concentration", required=True,
    help ="cyanide concentration")    
ap.add_argument("-ps", "--paper_sensor", required=False,
    help="folder name of the paper sensor")
ap.add_argument("-i", "--images", type = int, default = 5,
    help="number of images to be captured")
ap.add_argument("-cif", "--captured_images_dir", required=False,
    help="folder name of the images gathered")
ap.add_argument("-e", "--exposure", type = int, default = 100,
    help="exposure value of the camera")
ap.add_argument("-in", "--interval", type = int, default = 30,
    help="exposure value of the camera")
ap.add_argument("-r", "--reso", default = "H",
    help="resolution of the image captured( 'L' for low, 'M' for medium, 'H' for high)")
args = vars(ap.parse_args())
print(args)


hiRes = (2592, 1952)
medRes = (2048,1080)
lowRes = (640,480)

if args["reso"] == "H":
    res = hiRes
elif args["reso"] == "M":
    res = medRes
elif args["reso"] == "L":
    res = lowRes

camera = PiCamera()

###Set camera parameters
camera.resolution = res
camera.iso = args['exposure']
camera.shutter_speed = camera.exposure_speed
camera.rotation = 180
camera.brightness = 50
camera.saturation = 0
camera.contrast = 0
sleep(3)

###DATA
######CAPTURED_IMAGES
#########PAPER_SENSOR
############CN_CONCENTRATION
###############TIMESTAMP
##################IMAGES.PNG


FILENAME = args['filename']
concentration = args ['concentration']
interval = args['interval']
img_num = args['images']

DATA_DIR = "DATA"
CAPTURED_IMAGES_DIR= args["captured_images_dir"]
PAPER_SENSOR_DIR = args['paper_sensor']
CN_CONCENTRATION_DIR = args["concentration"]

now = dt.datetime.now()
timestamp =  now.strftime("%Y-%m-%d-%H-%M")
TIME_DIR= str(timestamp) +','+str(args["exposure"])

CD = os.getcwd()
print(CD)
backCD =os.path.normpath(os.getcwd() + os.sep + os.pardir)
print(backCD)


DATA_PATH = os.path.join(backCD,DATA_DIR)
print(DATA_PATH)
CAPTURED_IMAGES_PATH = os.path.join(DATA_PATH,CAPTURED_IMAGES_DIR)
PAPER_SENSOR_PATH = os.path.join(CAPTURED_IMAGES_PATH, PAPER_SENSOR_DIR)
CN_CONCENTRATION_PATH = os.path.join(PAPER_SENSOR_PATH,CN_CONCENTRATION_DIR)
IMG_PATH = os.path.join(CN_CONCENTRATION_DIR,TIME_DIR,'')


if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)

if not os.path.exists(CAPTURED_IMAGES_PATH):
    os.mkdir(CAPTURED_IMAGES_PATH)


if not os.path.exists(PAPER_SENSOR_PATH):
    os.mkdir(PAPER_SENSOR_PATH)


if not os.path.exists(CN_CONCENTRATION_PATH):
    os.mkdir(CN_CONCENTRATION_PATH)

if not os.path.exists(IMG_PATH):
    os.mkdir(IMG_PATH)





### Function for taking an image

def take_img(i,fill,IMG_PATH,display= False , save_img = False ):

    ###Capturing to an OpenCV object
    # camera.start_preview()
    
    now_2 = dt.datetime.now()
    timestamp_2 =  now_2.strftime("%S")
    IMG_PATH =  IMG_PATH +str(i).zfill(fill) +','+ args["concentration"]+','+str(timestamp_2)+','+str(i*30)+',seconds.png'
    img = np.empty((1952*2592*3), dtype=np.uint8)
    img = img.reshape((1952,2592,3))

    if display:
        cv2.imshow('IMG', img)
        cv2.waitKey(0)

    if save_img:
        cv2.imwrite(IMG_PATH,img)

    # camera.stop_preview()
    return img 

### Function for taking multiple images automatically

def take_multi_img(img_num, time_interval,IMG_PATH, display = False, save_img = False, prev= False):
    if prev: 
        camera.start_preview()
    i = 0
    images = []
    img_1 = take_img(i,3,IMG_PATH,True, True)
    images.append(img_1)
    start = dt.datetime.now()
    total_time = img_num * time_interval
    interval = time_interval
    previous_time = dt.datetime.now()
    img = np.empty((1952* 2592* 3), dtype=np.uint8)
    while (dt.datetime.now()- start).seconds < total_time:

        if(dt.datetime.now()-previous_time).seconds >= interval:
            previous_time = dt.datetime.now()
            print("okay")
            img = take_img(i,2,IMG_PATH,True, True)
            i += 1
            images.append(img)
    
    if prev:
        camera.stop_preview()
    
    print("Done")
    return images



if __name__ == '__main__':
    camera.start_preview()
    camera.annotate_text = "Press ENTER to start the image acquisition for the 1st image or CTRL-D to Quit"
    try:
        input("Press ENTER to start the image acquisition")
    except SyntaxError:
        pass
    img = take_img(0,3,IMG_PATH,display=False, save_img = True)
    camera.stop_preview()

    sleep(3)

    try:
        input("Press ENTER to start the image acquisition")
    except SyntaxError:
        pass

    camera.start_preview()
    take_multi_img(img_num,interval,IMG_PATH, False,True,False)
    camera.stop_preview()
    camera.close()
    print("Done")


