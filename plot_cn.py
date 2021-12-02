
import numpy as np
import matplotlib.pyplot as plt


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




def scatter_plotR(sorted_data, ppm_values_str,data_path):
    fig_r = plt.figure()
    axes_r = fig_r.add_subplot(111)

    #plot the RGB_Mean Intensity of the paper sensor that was taken
    axes_r.scatter(sorted_data[:,0].astype(float),sorted_data[:,2].astype(float),color='red', marker='x', label = "Red")
       # for a,b in zip(sorted_data[:,0].astype(float).astype(int), sorted_data[:,2].astype(float).astype(int)): 
    #     plt.text(a+1, b+1, str(b))
    
    # for aa,bb in zip(sorted_data[:,0].astype(float).astype(int), sorted_data[:,3].astype(float).astype(int)): 
    #     plt.text(aa + 1, bb+1, str(bb))

    # for aaa,bbb in zip(sorted_data[:,0].astype(float).astype(int), sorted_data[:,4].astype(float).astype(int)): 
    #     plt.text(aaa+1, bbb+1, str(bbb))
    axes_r.set_ylabel('Mean Pixel Intensity')
    axes_r.set_xlabel('Cyanide Concentration (PPM)')
    axes_r.set_title("Red-Channel")
    axes_r.set_xlim(-20, 120)
    axes_r.set_ylim(0, 270)
    axes_r.legend()
    # plt.show()
    fig_r.savefig(data_path+'/RPLOT1.png')
    print("Saved Plot")

def scatter_plotG(sorted_data, ppm_values_str,data_path):
    fig_g = plt.figure()
    axes_g = fig_g.add_subplot(111)

    #plot the RGB_Mean Intensity of the paper sensor that was taken
    axes_g.scatter(sorted_data[:,0].astype(float),sorted_data[:,3].astype(float),color='green', marker='x', label = 'Green')
    # for a,b in zip(sorted_data[:,0].astype(float).astype(int), sorted_data[:,2].astype(float).astype(int)): 
    #     plt.text(a+1, b+1, str(b))
    
    # for aa,bb in zip(sorted_data[:,0].astype(float).astype(int), sorted_data[:,3].astype(float).astype(int)): 
    #     plt.text(aa + 1, bb+1, str(bb))

    # for aaa,bbb in zip(sorted_data[:,0].astype(float).astype(int), sorted_data[:,4].astype(float).astype(int)): 
    #     plt.text(aaa+1, bbb+1, str(bbb))
    axes_g.set_ylabel('Mean Pixel Intensity')
    axes_g.set_xlabel('Cyanide Concentration (PPM)')
    axes_g.set_title("Green-Channel")
    axes_g.set_xlim(-20, 120)
    axes_g.set_ylim(0, 270)
    axes_g.legend()
    # plt.show()
    fig_g.savefig(data_path+'/GPLOT1.png')
    print("Saved Plot")

def scatter_plotB(sorted_data, ppm_values_str,data_path):
    fig_b = plt.figure()
    axes_b = fig_b.add_subplot(111)

    #plot the RGB_Mean Intensity of the paper sensor that was taken
    axes_b.scatter(sorted_data[:,0].astype(float),sorted_data[:,4].astype(float),color='blue', marker='x', label= 'Blue')
    # for a,b in zip(sorted_data[:,0].astype(float).astype(int), sorted_data[:,2].astype(float).astype(int)): 
    #     plt.text(a+1, b+1, str(b))
    
    # for aa,bb in zip(sorted_data[:,0].astype(float).astype(int), sorted_data[:,3].astype(float).astype(int)): 
    #     plt.text(aa + 1, bb+1, str(bb))

    # for aaa,bbb in zip(sorted_data[:,0].astype(float).astype(int), sorted_data[:,4].astype(float).astype(int)): 
    #     plt.text(aaa+1, bbb+1, str(bbb))
    axes_b.set_ylabel('Mean Pixel Intensity')
    axes_b.set_xlabel('Cyanide Concentration (PPM)')
    axes_b.set_title("Blue-Channel")
    axes_b.set_xlim(-20, 120)
    axes_b.set_ylim(0, 270)
    axes_b.legend()
    # plt.show()
    fig_b.savefig(data_path+'/BPLOT1.png')
    print("Saved Plot")
