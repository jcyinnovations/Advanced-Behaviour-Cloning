import cv2
import numpy as np
from time import time
import pickle
import math
from time import time

ip_kernel_2x2 = np.ones((2,2),np.uint8)
ip_kernel_3x3 = np.ones((3,3),np.uint8)

##################################
# Format timestamp
##################################
def process_time_str(start, end):
    etime = end - start
    return "{0:.0f}:{1:.0f} minutes".format((etime-(etime%60))/60, round(etime%60))


##############################################################################################
# Crop an image array and resize to VGG16 ImageNet minimum dimensions (48,48)
# Using OpenCV for resize operation in both trainer and drive algorithms
##############################################################################################
def get_viewport(image, x=0,y=65, w=320,h=80, size=(80,20)):
    #CENTER: 
    x1, y1 = x+w, y+h
    img = image[y:y1, x:x1]
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img

    
##########################################################################
#Increase Image contrast
##########################################################################
def image_contrast(img, phi=1, theta=1, maxIntensity=255.0):
    img = (maxIntensity/phi)*(img/(maxIntensity/theta))**2
    return img.astype("uint8")

    
##########################################################################
# Increase Image brightness
##########################################################################
def image_brightness(img, phi=1, theta=1, maxIntensity=255.0):
    img = (maxIntensity/phi)*(img/(maxIntensity/theta))**0.5
    return img.astype("uint8")

    
# Sketch edges of image and return a black and white image in 1/3 channels
# Applies a gradient filter with a default 2x2 kernel.
# Assumes a color (3 channel) input image
##########################################################################
def image_gradient(img, channels=3):
    newImg = cv2.morphologyEx(grayscale(img), cv2.MORPH_GRADIENT, ip_kernel_2x2, iterations=1)
    if channels == 3:
        newImg = cv2.cvtColor(newImg, cv2.COLOR_GRAY2BGR)
    return newImg

    
##########################################################################
# Sharpen images using a simple filter since a lot of images are blurred
##########################################################################
def image_sharpen(img):
    #Create the filter kernel
    kernel = np.zeros( (9,9), np.float32)
    kernel[4,4] = 2.0
    boxFilter = np.ones( (9,9), np.float32) / 81.0
    kernel = kernel - boxFilter
    #Apply the filter
    return cv2.filter2D(img, -1, kernel)

    
##########################################################################
# Flip an image horizontally/vertically
# openCV: True returns BGR, False returns RGB
##########################################################################
def image_flip(img, horizontal=1):
    image = cv2.flip(img, horizontal)
    return image


#####################################################
#Cleanup images using OpenCV Histogram Equalization
#####################################################
def image_preprocessing(image):
    #img = img.view('uint8')[:,:,::4]
    #Recast as uint8 to support OpenCV requirements
    img = image.astype("uint8")
    #Convert to YUV before equalization
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output

############################################
# Data-set augmentation functions compliments
# of Vivek Yadav (posted to CarND space)
############################################
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def transform_image(img,ang_range,shear_range,trans_range):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over. 
    A Random uniform distribution is used to generate different parameters for transformation
    '''
    # Rotation
    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])
    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2
    
    # Brightness 
    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)
    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))
    img = augment_brightness_camera_images(img)
    
    return img

#
# First preprocess each image with Histogram Equalization
# Then apply a sharpening filter to try to lift features from blurry images
#########################################################################################
def image_normalization(x, skip_hsv=False):
    start_time = time()
    for i in range(len(x)):
        x[i] = image_preprocessing(x[i])
        x[i] = image_sharpen(x[i])
        if not skip_hsv:
            x[i] = cv2.cvtColor(x[i], cv2.COLOR_BGR2HSV)
    end_time = time()
    print("Image Processing complete ", process_time_str(start_time, end_time))
    return x

#
#Basic data normalization for 8-bit images
#
def data_normalization(x, depth=255):
    return x/depth - 0.5

#
# Generate list of indicies for each sign type
# This supports data normalization where we need to add images to balance the dataset
# As well as subsequent class performance evaluation to see if there is skew in the model
# performance
###########################################################################################
def data_binning(imgs, labels, class_count, scale=1):
    #Bin the test data so each sign class can be tested for accuracy
    bins = []
    label_bins = []
    img_bins = []
    bin_count = []
    
    indicies = np.arange(len(labels))
    #For each class generate the bin contents
    for t in range(class_count):
        value = int(t * scale)
        
        if scale > 1:
            condition = np.logical_and(labels >= value, labels < value+scale)
        else:
            condition = labels == value
        idx_list = np.extract(condition, indicies)
        #print("Bin ", t, "count ", len(idx_list))
        bins.append(idx_list)
        label_bins.append(labels[idx_list])
        img_bins.append(imgs[idx_list])
        bin_count.append(len(idx_list))
    return bins, img_bins, label_bins, bin_count

#
# Balance the distribution of features across the classes by generating and removing selectively 
######################################################################################################
def balance_dataset(X_train, y_train, n_classes):
    #First generate the bins for training data and calculate the average per class
    bins, X_train_binned, y_train_binned, bin_count = data_binning(X_train, y_train, n_classes)
    average_count = round(np.mean(bin_count))

    #Now downsample the above average classes to average by randomly picking 
    #enough samples and removing them from the training set
    removal_list = None
    for i in range(n_classes):
        if bin_count[i] > average_count:
            removal_count = int(0.90 * (bin_count[i] - average_count) )
            #print("removing",removal_count, " from class ", i)
            #generate a list of random removals
            removal_idx = np.random.choice(bin_count[i], size=removal_count, replace=False)
            if removal_list is None:
                removal_list = bins[i][removal_idx.tolist()]
            else:
                removal_list = np.append(removal_list, bins[i][removal_idx.tolist()])

    #Now remove the targeted samples
    X_train = np.delete(X_train, removal_list, axis=0)
    y_train = np.delete(y_train, removal_list, axis=0)

    #Now Add new samples for under average classes
    for i in range(n_classes):
        if bin_count[i] < average_count:
            add_count = int( 1.10 * (average_count - bin_count[i]) )
            #pick images at random to dither 10 times
            src_img_count = int(round(add_count / 10))
            print("class ", i, "dither ", src_img_count)
            src_idx = np.random.choice(bin_count[i], size=src_img_count, replace=False)
            #print("sources", src_idx)
            #print("bin size", len(X_eval[i]))
            for src in range(src_img_count):
                nimg = X_train_binned[i][src_idx[src]]
                new_img_list = []
                for j in range(10):
                    dithered_img = transform_image(nimg,10,2,5)
                    new_img_list.append(dithered_img)

                #X_train = np.append(X_train, [nimg,nimg,nimg,nimg,nimg,nimg,nimg,nimg,nimg,nimg], axis=0)
                X_train = np.append(X_train, new_img_list, axis=0)
                y_train = np.append(y_train, [i,i,i,i,i,i,i,i,i,i])
    #Pickle the augmented data to save time
    pickle_dict = {'features': X_train, 'labels': y_train}
    pickle.dump(pickle_dict, open("./train_augmented.p", "wb"))
    return X_train, y_train


# Convert to grayscale
##########################################################
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

###########################################################    
# Canny Edge Detection
###########################################################
def canny(img, low_threshold, high_threshold=0):
    """Applies the Canny transform"""
    if high_threshold == 0:
        high_threshold = low_threshold * 3
    return cv2.Canny(img, low_threshold, high_threshold)

###########################################################
# Gaussian Blur
###########################################################
def gaussian_blur(img, kernel_size=3):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

###########################################################
# Masking the region of interest
###########################################################
def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


###########################################################
# Calculate line gradients
###########################################################
def line_params(x1,y1,x2,y2, xsize, ysize):
    """
    Calculate the parameters for a hough line (slope, intercept, and extrapolations)
    """
    #Generate m, c for line equation y=mx+c
    m = (y2 - y1)/(x2-x1)
    c = y1 - m*x1
    max_y = ysize
    if not math.isnan(1/m):
        max_x = (max_y-c)/m
    else:
        max_x = x2
        max_y = y2
        
    mid_y = ysize*0.6
    if not math.isnan(1/m):
        mid_x = (mid_y-c)/m
    else:
        mid_x = x2
        mid_y = y2
    return (m, c, mid_x, mid_y, max_x, max_y)

###############################################################################################
# Draw Lines based on a weighted average of the gradients for left and right.
# This is actually a crappy implementation. Draw Lines 2 is better
###############################################################################################
def draw_lines(img, lines, color=[0, 0, 255], thickness=5, filter_lines=True, min_slope=0.25):
    #Reject almost horizontal lines to try and cleanup last video
    xsize = img.shape[1] #Shape is used to determine where to extrapolate lines
    ysize = img.shape[0]
    rx_max = 0
    rx_mid = 0
    lx_max = 0 
    lx_mid = 0
    rm = 0
    lm = 0
    
    if filter_lines:
        for line in lines:
            for x1,y1,x2,y2 in line:
                params = line_params(x1,y1,x2,y2, xsize, ysize)
                if abs(params[0]) > min_slope:
                    if params[0] > 0:
                        #print("right", params)
                        rx_max += params[0] * params[4]
                        rx_mid += params[0] * params[2]
                        rm += params[0]
                    else:
                        #print("left", params)
                        lx_max += -1 * params[0] * params[4]
                        lx_mid += -1 * params[0] * params[2]
                        lm += -1 * params[0]
        #Calculate weighted average of coordinates to try and reject some of the noise
        y_mid = int(ysize/4)
        y_max = ysize
        if rm != 0 and not math.isnan(rx_max/rm):
            #rm = 1
            rx_max = int(rx_max / rm)
            rx_mid = int(rx_mid / rm)
            cv2.line(img,(rx_max,y_max),(rx_mid,y_mid), color, thickness)
        if lm != 0 and not math.isnan(lx_max/lm):
            #lm = 1
            lx_max = int(lx_max / lm)
            lx_mid = int(lx_mid / lm)
            cv2.line(img,(lx_max,y_max),(lx_mid,y_mid), color, thickness)
    else:
        for line in lines:
            for x1,y1,x2,y2 in line:
                params = line_params(x1,y1,x2,y2, xsize, ysize)
                if abs(params[0]) > min_slope:
                    cv2.line(img, (x1, y1), (x2, y2), color, thickness)

###############################################################################################
# Draw Lines by grouping them by gradient. Lines with similar gradient and zero crossing 
#  are assumed to be colinear and grouped together
###############################################################################################
def draw_lines2(img, lines, color=[0, 0, 255], thickness=5, filter_lines=True, min_slope=0.25):
    #Reject almost horizontal lines to try and cleanup last video
    xsize = img.shape[1] #Shape is used to determine where to extrapolate lines
    ysize = img.shape[0]
    rx_max = 0
    rx_mid = 0
    lx_max = 0 
    lx_mid = 0
    rm = 0
    lm = 0
    
    if i == 0:
        x_train = np.array([img])
    else:
        x_train = np.append(x_train,[img], axis=0)
    
    if filter_lines:
        for line in lines:
            for x1,y1,x2,y2 in line:
                m, c, mid_x, mid_y, max_x, max_y = line_params(x1,y1,x2,y2, xsize, ysize)
                if abs(params[0]) > min_slope:
                    if params[0] > 0:
                        #print("right", params)
                        rx_max += params[0] * params[4]
                        rx_mid += params[0] * params[2]
                        rm += params[0]
                    else:
                        #print("left", params)
                        lx_max += -1 * params[0] * params[4]
                        lx_mid += -1 * params[0] * params[2]

######################################################                  
# Apply Hough Transform
######################################################
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, filter_lines=True, min_slope=0.25):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    if lines is not None:
        draw_lines(line_img, lines, filter_lines=True, min_slope=0.25)
    return line_img

######################################################
# Overlay Hough lines on Image
######################################################
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)
