from skimage import exposure
import matplotlib.pyplot as plt
import argparse
import cv2
import os
import numpy as np 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True,
	help="path to the input source image")
ap.add_argument("-r", "--reference", required=True,
	help="path to the input reference image")
args = vars(ap.parse_args())



def get_image(image_path):
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

ref = args["reference"]
src = args["source"]
matched_path = "matched_folder"
os.mkdir(matched_path)
images = []
images_src = []
images_ref = []
src_paths = []
ref_paths = []
for image_ref in os.listdir(ref):
         
    # print('image_filename',image)
    image_number, ppm_value,sec,seconds,second = image_ref.split(',')
    image_ref_path = os.path.join(ref, image_ref)

    
    images_ref.append(get_image(image_ref_path))
    # img_src = get_image(image_ref_path)
    ref_paths.append(image_ref)


for image_src in os.listdir(src):
         
    # print('image_filename',image)
    image_number, ppm_value,sec,seconds,second = image_src.split(',')
    image_src_path = os.path.join(src, image_src)

    src_paths.append(image_src)
    images_src.append(get_image(image_src_path))
    # img_src = get_image(image_path)
    # image_paths.append(image_path)
    # ppm_values.append(ppm_value)
    # print(image_number)
    # print(image)
    # print(img)



for i in range(len(images_src)):

    save_path = matched_path +'\\'+src_paths[i]
    print(save_path)

    print("[INFO] performing histogram matching...")
    multi = True if images_src[i].shape[-1] > 1 else False
    matched = exposure.match_histograms(images_src[i], images_ref[i], multichannel=multi)

    # cv2.imshow("Source", img)
    # cv2.imshow("Reference", ref)
    # cv2.imshow("Matched", matched)
    # cv2.waitKey(0)
    cv2.imwrite(save_path,matched)

# show the output images
# cv2.imshow("Source", src)
# cv2.imshow("Reference", ref)
# cv2.imshow("Matched", matched)
# cv2.waitKey(0)