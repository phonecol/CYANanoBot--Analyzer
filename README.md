# CYANanoBot--Analyzer
Repository for CYANanoBot Paper Sensor Analyzer by Ian Benedict Pongcol


### Run Image Acquisition Program
$ python image_acquisition_program.py -cn {0ppm-100ppm} -ps {paper_sensor_name} -i {30-40} -cif {captured_images3} -e {100} -in {30} -r {H} 

$ python image_acquisition_program.py -cn 100ppm -ps paper_sensor_name -i 30 -cif captured_images3 -e 100 -in 30 -r H 


### Run Paper Sensor Detection Program
$ python paper_sensor_detection_program.py -id2 {000-30} -ps {paper_sensor_name} 
-cif {captured_images_subfolder} -nr {1-3}
$ python paper_sensor_detection_program.py -ps new_sensor -id2 000 -cif captured_images3 -nr 1 -si True


-DATA
--CAPTURED_IMAGES
---PAPER_SENSOR
----CN_CONCENTRATION
-----TIMESTAMP
------IMAGES.PNG

-DATA
--PAPER_SENSOR_DATA
---ROI_PAPER_SENSOR
---ROI2_PAPER_SENSOR



### Run Color Extraction Program
$ python color_extraction_program.py -id2 {000-30} -rf2 {ROI2_filename} -fn {filename}
$ python color_extraction_program.py -id2 000 -rf2 ROI2_newsensor_45min -fn newsensor_45min_data
-DATA
--PAPER_SENSOR_DATA
---model
---ROI_PAPER_SENSOR
---ROI2_PAPER_SENSOR
---ColorData
----IMG_ID
-----CSV
-----PLOTS
-----HISTOGRAMS
------HistogramsCSV
------HistogramsPNG


### Run Create Regression Model Program
$ python create_regression_model.py -id2 {000-30} -rf2 {ROI_paper_sensor} -fn {paper_sensor_data}
$ python create_regression_model.py -id2 000 -rf2 ROI2_newsensor_45min -fn new_sensor_data



