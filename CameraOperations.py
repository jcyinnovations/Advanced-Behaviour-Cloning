#################################################################
#################################################################
## Camera Operations: Assumes all images are RGB
#################################################################
#################################################################

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

#################################################################
# Plot a grid of labeled images
#################################################################
def show_grid(images, labels, cmap_mono="gray", ticks=None):
    cols = len(images)
    f, axes = plt.subplots(1, cols, figsize=(20,10))
    for img, label, axis in zip(images, labels, axes):
        axis.set_title(label)
        if ticks is not None:
            w = img.shape[1]
            h = img.shape[0]
            axis.set_yticks([t for t in range(0,h,ticks[1])])
            axis.set_xticks([t for t in range(0,w,ticks[0])])
            axis.grid(b=True,color='w')
        if len(img.shape) == 3:
            axis.imshow(img)
        else:
            axis.imshow(img, cmap=cmap_mono)


#################################################################
# Convert BGR images to RGB for display
#################################################################
def disp(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


#################################################################
# Load images into an array using a list of names of a pattern
#################################################################
def load_images_from_folder(folder, images=None, name_pattern=None):
    image_array = None
    if images is None and name_pattern is not None:
        images = glob.glob("{0}/{1}".format(folder,name_pattern))
    else:
        return None

    for fname in images:
        img = cv2.imread(fname)
        if image_array is None:
            image_array = np.array([img])
        else:
            image_array = np.append(image_array, [img], axis=0)
    return image_array


#################################################################
# Calibrate camera with chessboard images
#################################################################
def calibrate_camera(images, counts=(9,6)):
    objpoints = [] # 3D points in real space
    imgpoints = [] # 2D points in image space

    objp = np.zeros( (counts[0]*counts[1], 3), np.float32 )
    objp[:,:2] = np.mgrid[0:counts[0], 0:counts[1]].T.reshape(-1, 2) #Generate x,y coordinates

    for i in range(len(images)):
        gray = cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, counts, None)
        if ret==True:
            imgpoints.append(corners)
            objpoints.append(objp)
            #img_corners = cv2.drawChessboardCorners(image, counts, corners, ret)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    return mtx, dist


#################################################################
# Calibrate camera with chessboard images
#################################################################
def calibrate_camera_from_folder(folder, images=None, name_pattern=None, counts=(9,6)):
    if images is None and name_pattern is not None:
        images = glob.glob("{0}/{1}".format(folder,name_pattern))
    else:
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints = [] # 3D points in real space
    imgpoints = [] # 2D points in image space
    objp = np.zeros( (counts[0]*counts[1], 3), np.float32 )
    objp[:,:2] = np.mgrid[0:counts[0], 0:counts[1]].T.reshape(-1, 2) #Generate x,y coordinates

    for fname in images:
        #img = mpimg.imread(fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, counts, None)
        if ret==True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
            #img_corners = cv2.drawChessboardCorners(img, counts, corners2, ret)
            #plt.figure()
            #plt.imshow(img_corners)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        print("Image", images[i], "Error",error)
        mean_error += error
    print("Total error: ", mean_error/len(objpoints))
    return mtx, dist


#################################################################
# Correct camera image for calibrated distortion
#################################################################
def correct_distortion(img, mtx, dist):
    un_dst = cv2.undistort(img, mtx, dist, None, mtx)
    return un_dst


#################################################################
#Perspective Transform: extract lanes and view from above
#################################################################
def perspective_transform(image, viewport=[[0,0],[0,0],[0,0],[0,0]], offset=80, reverse=False):
    w, h = image.shape[1], image.shape[0]
    src = np.float32(viewport)
    #dst = np.float32([[offset, offset], [w-offset, offset], [w-offset, h-offset], [offset, h-offset]])
    #dst = np.float32([[offset,0],[w-offset,0],[w-offset,h+offset],[offset,h+offset]])
    dst = np.float32([[0,0],[w-offset,0],[w-offset,h],[0,h]])
    M = cv2.getPerspectiveTransform(src, dst)
    if reverse:
        M = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(image, M, (w-offset,h), flags=cv2.INTER_LINEAR)
    return warped

#################################################################
# Sobel Thresholding in x, y and both directions
# 'Both' gives the magnitude of the gradient
#################################################################
def sobel_threshold(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    #Check if single channel before converting to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    if orient == 'x':
        sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    if orient == 'both':
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        sobel = np.sqrt( np.square(sobelx) + np.square(sobely) )
    scaled_sobel = np.uint8(255*sobel/np.max(sobel))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sxbinary


#################################################################
# Gradient Gradient Direction
#################################################################
def gradient_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    gradient = np.arctan2(sobely, sobelx)
    sxbinary = np.zeros_like(gradient).astype("uint8")
    sxbinary[(gradient >= thresh[0]) & (gradient <= thresh[1])] = 1
    return sxbinary

####################################################################
# Combined Sobel plus Gradient Magnitude and direction Thresholding
####################################################################
def sobel_and_gradient(img, sobel_kernel=3, sobel_thresh=(80, 160), gradient_thresh=(0, np.pi/2)):
    gradx = sobel_threshold(img, orient='x', sobel_kernel=sobel_kernel, thresh=sobel_thresh)
    grady = sobel_threshold(img, orient='y', sobel_kernel=sobel_kernel, thresh=sobel_thresh)
    mag_binary = sobel_threshold(img, orient='both', sobel_kernel=sobel_kernel, thresh=sobel_thresh)
    dir_binary = gradient_threshold(img, sobel_kernel=sobel_kernel, thresh=gradient_thresh)
    combined = np.zeros_like(dir_binary).astype("uint8")
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined

#################################################################
# Thresholding on S (HLS) channel
#################################################################
def threshold_hls(image, thresh=(0,255)):
    if len(image.shape) == 3:
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        S = hls[:,:,2]
    else:
        S = image
    binary_s = np.zeros_like(S)
    binary_s[(S > thresh[0]) & (S <= thresh[1])] = 1
    return binary_s

#################################################################
# Thresholding on R (RGB) channels
#################################################################
def threshold_rgb(image, thresh=(0,255)):
    if len(image.shape) == 3:
        R = image[:,:,2]
    else:
        R = image
    binary_r = np.zeros_like(R)
    binary_r[(R > thresh[0]) & (R <= thresh[1])] = 1
    return binary_r

#################################################################
# Simple Thresholding on combined S (HLS) and R (RGB) channels
#################################################################
def combined_SR_threshold(image, sobel_thresh=(20, 100), s_thresh=(190,240), sobel_kernel=15):
    sxbinary = sobel_threshold(image[:,:,0], orient='x', sobel_kernel=sobel_kernel, thresh=sobel_thresh)
    s_binary = threshold_hls(image, thresh=s_thresh)
    color_binary = 255*np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary, color_binary


##################################################################
# Thresholding on combined Luminosity (HLS) and Red (RGB) channels
##################################################################
def sobel_LR_threshold(image, sobel_kernel=15, sobel_thresh=(30, 150)):
    r = image[:,:,0]
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    l = hls[:,:,1]
    sobel_r = sobel_threshold(r, orient='x', sobel_kernel=sobel_kernel, thresh=sobel_thresh)
    sobel_l = sobel_threshold(l, orient='x', sobel_kernel=sobel_kernel, thresh=sobel_thresh)
    combined = np.zeros_like(sobel_r)
    combined[((sobel_r == 1) | (sobel_l == 1))] = 1
    return combined

##############################################################################
# Thresholding on combined Luminosity, Saturation (HLS) and Red (RGB) channels
##############################################################################
def sobel_LSR_threshold(image, sobel_kernel=15, sobel_thresh=(30, 150)):
    r = image[:,:,2]
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    l = hls[:,:,1]
    s = hls[:,:,2]
    sobel_r = sobel_threshold(r, orient='x', sobel_kernel=sobel_kernel, thresh=sobel_thresh)
    sobel_l = sobel_threshold(l, orient='x', sobel_kernel=sobel_kernel, thresh=sobel_thresh)
    sobel_s = sobel_threshold(s, orient='x', sobel_kernel=sobel_kernel, thresh=sobel_thresh)
    #t_r = threshold_rgb(r, thresh=(220,255))
    combined_slr = np.zeros_like(sobel_r)
    combined_slr[((sobel_r == 1) | (sobel_l == 1) | (sobel_s == 1))] = 1 # | (t_r == 1)
    return combined_slr

##############################################################################
# Gradient & Threshold on Luminosity and Saturation (HLS) channels
# Luminosity performs better in shadow whereas Saturation performs better
# generally.
##############################################################################
def sobel_LS_threshold(image, sobel_kernel=15, sobel_thresh=(30, 150), gradient_thresh=(0, np.pi/2)):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    l = hls[:,:,1]
    s = hls[:,:,2]
    s = sobel_and_gradient(s, sobel_kernel=sobel_kernel, sobel_thresh=sobel_thresh, gradient_thresh=gradient_thresh)
    l = sobel_and_gradient(l, sobel_kernel=sobel_kernel, sobel_thresh=sobel_thresh, gradient_thresh=gradient_thresh)

    combined = np.zeros_like(s).astype("uint8")
    combined[((s == 1) | (l == 1))] = 1
    return combined
