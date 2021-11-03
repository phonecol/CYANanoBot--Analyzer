import subprocess
print("Paper sensor detection and cropping program")
# subprocess.run(['python', 'openfolders_multiple.py', '-fn', 'new_sensor','-id2','10', '-rf1','ROI_newsensor', '-rf2', 'ROI2_newsensor', '-cif','captured_images3','-cisf', 'data_gathering_newsensor', '-nr','3'])

# python openfolders_multiple.py -fn new_sensor -id2 000 -rf1 ROI_newsensor -rf2 ROI2_newsensor -cif captured_images3 -cisf data_gathering_newsensor -fn newsensor_45min
print("Done")

print("Color Extraction Program")
subprocess.run(['python', 'color_extraction.py','-id2','11', '-rf2', 'ROI2_newsensor', '-fn', 'new_sensor'])
# python color_extraction.py -id2 000 -rf2 ROI2_newsensor_45min
print("Done")


print("Color")

