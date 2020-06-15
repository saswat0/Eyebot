# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 17:51:57 2019

@author: hughd

press 'r' to toggle rectangle
press 'c' to toggle contours and centroid pixel
press 'e' to toggle ellipse
Use W,A,S,D keys to move green pixel marking the location of the true pupil centre
"""

#%% import libraries and define functions
# need OpenCV version 3.4.7.28
#pip install opencv-python==3.4.7.28
import matplotlib.pyplot as plt
import numpy as np
import cv2

def nthsmallest(x,n):
    '''find the nth smallest value of an array'''
    n = int(n) # for use as index
    flat = x.flatten()
    flat.sort()
    if n < len(flat):
        out = flat[n]
    return out

def on_trackbar_eye(val):
    ''' Callback function for changing the threshold value'''
    global thresh # change the global variable for threshold value
    thresh = val

def on_trackbar_Leye(val):
    ''' Callback function for changing the threshold value'''
    global Lthresh # change the global variable for threshold value
    Lthresh = val

def on_trackbar_Reye(val):
    ''' Callback function for changing the threshold value'''
    global Rthresh # change the global variable for threshold value
    Rthresh = val

def on_trackbar_PR(val):
    ''' Callback function for changing the threshold value'''
    global PRthresh # change the global variable for threshold value
    PRthresh = val

def on_trackbar_LPR(val):
    ''' Callback function for changing the threshold value'''
    global LPRthresh # change the global variable for threshold value
    LPRthresh = val

def on_trackbar_RPR(val):
    ''' Callback function for changing the threshold value'''
    global RPRthresh # change the global variable for threshold value
    RPRthresh = val

def threshold(img, thresh, blur):
    '''Function to threshold eye image using opening and bluring
    This version has the thresh value be the value of the nth darkest value'''
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if blur:
        img_gray = cv2.GaussianBlur(img_gray, (7, 7), 0)  #blur before thresholding to remove noise
    # offset values so they start at 0 this way only relative brightness matters
    img_gray = img_gray - np.amin(img_gray)  # redundant
    _, imgthreshold = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY_INV)
    #imgthresholdold = 1*imgthreshold # duplicate the one before opening for comparison
    #open_kernel = np.ones((2,2))
    #imgthreshold = cv2.morphologyEx(imgthreshold, cv2.MORPH_OPEN, open_kernel)
    #imgthreshold = cv2.medianBlur(imgthreshold, 5)
    _, imgcontours, _ = cv2.findContours(imgthreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    imgcontours = sorted(imgcontours, key=lambda x: cv2.contourArea(x), reverse=True)
    return imgthreshold, imgcontours, img_gray


def threshold(img, thresh, blur):
   '''Function to threshold eye image using opening and bluring
   This version has the thresh value be the value of the nth darkest value'''
   img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   if blur:
       img_gray = cv2.GaussianBlur(img_gray, (7, 7), 0)  #blur before thresholding to remove noise
   # offset values so they start at 0 this way only relative brightness matters
   img_gray = img_gray - np.amin(img_gray)  # redundant
   if thresh >= len(img_gray.flatten()):
       thresh = len(img_gray.flatten())/2 # top stop crash when thresh is greater than num of pixels
   localthresh = nthsmallest(img_gray, thresh)
   _, imgthreshold = cv2.threshold(img_gray, localthresh, 255, cv2.THRESH_BINARY_INV)
   imgthresholdold = 1*imgthreshold # duplicate the one before opening for comparison
   open_kernel = np.ones((2,2))
   imgthreshold = cv2.morphologyEx(imgthreshold, cv2.MORPH_OPEN, open_kernel)
   imgthreshold = cv2.medianBlur(imgthreshold, 5)
   _, imgcontours, _ = cv2.findContours(imgthreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   imgcontours = sorted(imgcontours, key=lambda x: cv2.contourArea(x), reverse=True)
   return imgthreshold, imgcontours, img_gray

def findcentroid(img):
    'For a thresholded img, this ouputs centroid of non zero values'
    indices = np.asarray(np.where(img >0))
    centroid = np.transpose(np.mean(indices,axis=1))
    centroid = np.flip(centroid) #convert to x,y coords with flip
    return centroid

def fitrectangle(corners):
    # Fit a rectangle to corner points
    x = corners[:,0]
    y = corners[:,1]
    xmin = np.amin(x)
    #x2 = np.amax(x)
    ymin = np.amin(y)
    #y2 = np.amax(y)
    xrange = np.ptp(x) # range of x and y
    yrange = np.ptp(y)
    x1 = np.mean([z for z in x if z <= (xmin+xrange/2)])
    x2 = np.mean([z for z in x if z > (xmin+xrange/2)])
    y1 = np.mean([z for z in y if z <= (ymin+yrange/2)])
    y2 = np.mean([z for z in y if z > (ymin+yrange/2)])
    rectangle = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
    rect_corners = np.array([x1,x2,y1,y2])
    return rectangle, rect_corners

def fitrectangle2(corners1):
    ''' Fit rectangle to points
    function only works when points are given in right order'''
    # Fit a rectangle to corner points
    npoints = corners1.shape[0]
    pointspside = int((npoints-4)/4 + 2)
    corners = np.vstack([corners1, corners1[0,:]]) # repeat first element at end
    #find top and bottom y coords
    y1 = np.mean(corners[0*(pointspside-1):0*(pointspside-1)+pointspside, 1])
    y2 = np.mean(corners[2*(pointspside-1):2*(pointspside-1)+pointspside, 1])
    x1 = np.mean(corners[1*(pointspside-1):1*(pointspside-1)+pointspside, 0])
    x2 = np.mean(corners[3*(pointspside-1):3*(pointspside-1)+pointspside, 0])
    rectangle = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
    rect_corners = np.array([x1,x2,y1,y2])
    return rectangle, rect_corners

def findgaze(centroid, rect_corners, screen_img):
    '''Function to map pupil position to screnn rectangle position
    NB: returns nan if centroid is nan'''
    x1,x2,y1,y2 = rect_corners
    h = abs(y2-y1)
    w = abs(x2-x1)
    sh = screen_img.shape[0]
    sw = screen_img.shape[1]
    vscale = sh/h
    hscale = sw/w
    xc, yc = centroid
    xg = abs((xc-x1)*hscale)
    yg = abs((yc-y1)*vscale)
    gaze = np.array([xg,yg])
    return gaze

def rectpoints(img, npointspside, offset):
    '''plot npointspside pips on each side of a rectangel within and image,
    where offsets is how many pixels to offset the rectangle from teh image
    border'''
    x = np.linspace(0,1,npointspside) # x and y coords of pips
    y = np.linspace(0,1,npointspside)
    x = x.reshape(len(x),1)
    y = y.reshape(len(y),1)
    T = 0*x + 0 # x or y coord of side of rect
    B = 0*x + 1
    pips = np.vstack([np.hstack([x,T]),np.hstack([x,B]),np.hstack([B,y]),np.hstack([T,y])])
    xrectside = img.shape[1] - 2*offset
    yrectside = img.shape[0] - 2*offset
    pips[:,0] = pips[:,0]*xrectside +offset
    pips[:,1] = pips[:,1]*yrectside +offset
    pips = pips.astype(int)
    for a in range(0,pips.shape[0]):
        x,y = pips[a,:]
        cv2.circle(callib_image, (x,y), 5, (255,255,255), thickness=5, lineType=8, shift=0)
    return pips

def findBrightestRegion(img):
    '''Pick out the brightest region of a grayscale image
    p is the proportion of the image that should be selected'''
    #imsize = img.shape[0]*img.shape[1]
    # threshold so only brightest pixels remain
    #n = int(imsize*p)
    #thresh = nthsmallest(img,n)-1 # -1 because thresh_img = 0 when thresh = 255
    thresh = np.amax(img)-2
    global thresh_img
    _, thresh_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    kernel = np.ones((4,4))
    thresh_imgN = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)
    if np.sum(thresh_imgN) > 5: # stops crash when threshimg = 0 everywhere
        thresh_img = thresh_imgN
    indices = np.transpose(np.asarray(np.where(thresh_img >0)))
    coords = np.flip(indices)
    x = np.amin(coords[:,0])
    y = np.amin(coords[:,1])
    w = np.amax(coords[:,0])-x
    h = np.amax(coords[:,1])-y
    brightest_region = np.array([x,y,w,h]).reshape(1,4)
    #BRC = findcentroid(thresh_img) # centroid of brightest region
    return brightest_region

#%%Define Variables
'''Parameters that can be changed'''
single_eye = False # Boolean for single eye mode when in light mode
thresh = 60# initial threshold values for finding pupil of leftr and right eye
PRthresh = 60 # initial threshold values for refining pupil selection
frame_big_size = (1280,'?')

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

thresh_window = 'Image' # put the slider in the threshold window
slider_max = 255 # max slider value
#create trackbars to change pupil finding threshold
cv2.namedWindow(thresh_window)
thresh_slider = 'Pupil finder threshold %d' % slider_max
cv2.createTrackbar(thresh_slider, thresh_window , thresh, slider_max, on_trackbar_eye)
#create trackbars to change pupil refining threshold
cv2.namedWindow(thresh_window)
PR_slider = 'Pupil refiner threshold %d' % slider_max
cv2.createTrackbar(PR_slider, thresh_window , PRthresh, slider_max, on_trackbar_PR)

open_kernel = np.ones((2,2)) # kernal for opening thresholded pupil image

#import image
image = plt.imread(r'C:\Users\jason\OneDrive - The University of Nottingham\4th Year\Image Processing Files\Mini Project\Resolution Image.jpg')
#image = plt.imread('E:/Main Folders/Documents/Work/Fourth Year/Imaging/Eye tracking project/images/Looking towards nose/30N.png')
image = np.flip(image,axis=2) # convert to cv2's BGR
#image = np.flip(image,axis=0)
image = cv2.resize(image,(480,640))
image = leye_original3

frame_scale = int(frame_big_size[0]/image.shape[1]) # factor to scale up image
frame_big_size = (image.shape[1]*frame_scale,image.shape[0]*frame_scale)

dot = [int(frame_big_size[0]/2),int(frame_big_size[1]/2)]
manual_pup_centre = np.array([np.nan,np.nan])
show_contours = True
show_ellipse = True
show_rectangle = True
#%% Begin the main loop
while True:
    frame = image*1

    frame_original = frame*1 # copy the original frame
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = np.array([0,0,frame.shape[1],frame.shape[0]]).reshape(1,4)


    ellipse = np.zeros(5) #defined incase no ellipses, eyes, etc. found
    ellipse[:] = np.nan

    leye = np.ones((frame.shape[0],frame.shape[1],frame.shape[2]))*255  # so that leye can be defined if no eyes are found
    reye = np.ones((frame.shape[0],frame.shape[1],frame.shape[2]))*255
    lellipse = ((np.nan,np.nan),(np.nan,np.nan),np.nan) #so nan ellipse can fill
    rellipse = ((np.nan,np.nan),(np.nan,np.nan),np.nan) #array when no ellipse drawn



    #threshold eye without blur to select pupil without branching shadows
    thresholdOrig, contours, frame_grayOrig = threshold(frame, thresh, False)

    #Take region in largest contour of pupil and process it to refine pupil selection
    for cnt in contours:
        #define pupil bounding box
        pupborder = 1 # extend bounding rect to enclose more of pupil
        (x, y, w, h) = cv2.boundingRect(cnt)
        x = abs(x - pupborder)
        y = abs(y - pupborder)
        w = w + pupborder
        h = h + pupborder
        pups = np.array([x,y,w,h])

        #pupil = cv2.drawContours(leye, [cnt], -1, (0, 255, 255), 1)
        #if cnt.shape[0]>=5: #if at least 5 points on contour
            #lellipse = cv2.fitEllipse(cnt)
            #cv2.ellipse(leye, lellipse, (255,0, 255), 1, cv2.LINE_AA)

        #Take a new threshold of the pupil selection of eye image with blurring
        eyeblur = cv2.GaussianBlur(frame, (7, 7), 0) # blur values depend on iamge resolution
        pupblur = eyeblur[y:y+h,x:x+w,:] # take pupil region
        pup = frame[y:y+h,x:x+w,:]
        # find and draw the new contour in the refined selection
        pup_threshold, contoursN, pup_grayblur = threshold(pupblur, PRthresh, False) # threshold at lowest value
        #print('dfs')

        for cnt in contoursN:
            if show_contours:
                pupil = cv2.drawContours(pup, [cnt], -1, (0, 255, 255), 1)
            # fit an ellipse to the countour
            if cnt.shape[0]>=5: #if at least 5 points on contour
                ellipse = cv2.fitEllipse(cnt)
                if show_ellipse:
                    cv2.ellipse(pup, ellipse, (255,0, 255), 1, cv2.LINE_AA)
                #Convert ellipse parameters to an array for storing
                ellipse=np.hstack([ellipse[0],ellipse[1],ellipse[2]])
            break # for only the largest contour

        break #for only largest contour



    # Find centroid of pupil
    if np.sum(pup_threshold)>0:    # only if there are non zero values in threshold image
        centroid = findcentroid(pup_threshold)
        cent_coord = centroid.astype(int) #convert to integer for plotting
        if show_contours:
            pup[cent_coord[1],cent_coord[0]]=(0,0,255) # plot red pixel at centre of pupil
        #Alternatively plot circle at centre of pupil
        #cv2.circle(eye, (cent_coord[1],cent_coord[0]), 1, (0,0,255), thickness=1, lineType=8, shift=0)

    #Plot pixel at centre of fitted ellipse
    '''if not np.isnan(ellipse).any():    # only if there is an ellipse
        centroid = findcentroid(threshold) #ellipse[0:2]
        cent_coord = centroid.astype(int) #convert to integer for plotting
        eye[cent_coord[1],cent_coord[0]]=(0,0,255) # plot red pixel at centre of eye
        #Alternatively plot circle at centre of pupil
        #cv2.circle(eye, (cent_coord[1],cent_coord[0]), 1, (0,0,255), thickness=1, lineType=8, shift=0)'''

    #plot rectangles around pupil and eye
    if show_rectangle:
        px, py, pw, ph = pups
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 0, 255), 1)


    # enlarge image to sub pixel shift the manually placed dot
    frame_big = cv2.resize(frame,frame_big_size)
    # Plot dot for manual pupil location
    cv2.circle(frame_big, (dot[0],dot[1]), 2, (0,255,0), thickness=1, lineType=8, shift=0)

    #convert dot position on large image to position on original
    xscale = frame.shape[1]/frame_big.shape[1]
    yscale = frame.shape[0]/frame_big.shape[0] # both scales should be the same
    manual_pup_centre[0] = dot[0]*xscale
    manual_pup_centre[1] = dot[1]*yscale
    # convert manual pupil position error (estimated pm 1 pixel in big image)
    vectorUC = np.array([xscale,yscale])
    scalarUC = np.linalg.norm(vectorUC)

    # Convert coordinates so they are all relative to camera
    centroid = pups[0:2] + centroid
    #pupil ellipse centres
    ellipse[0:2] = pups[0:2] + ellipse[0:2]

    # calculate the vector and scalar distance from the real centre in pixels
    Vcentroid_diff = centroid - manual_pup_centre
    Vellipse_diff = ellipse[0:2]*1 - manual_pup_centre
    Scentroid_diff = np.linalg.norm(Vcentroid_diff)
    Sellipse_diff = np.linalg.norm(Vellipse_diff)

    # Normalise:convert these distances and there errors to units of the width
    # of the original image, i.e. divide by width of image
    NVcentroid_diff = Vcentroid_diff/image.shape[1]
    NVellipse_diff = Vellipse_diff/image.shape[1]
    NScentroid_diff = Scentroid_diff/image.shape[1]
    NSellipse_diff = Sellipse_diff/image.shape[1]
    NvectorUC = vectorUC/image.shape[1]
    NscalarUC = scalarUC/image.shape[1]

    cv2.imshow("Image", frame_big) # Display full frame

    key = cv2.waitKey(1)
    if key == 27: # press Esc to quit
        break
    if key == ord('w'):
        dot[1] = dot[1]-1
    if key == ord('s'):
        dot[1] = dot[1]+1
    if key == ord('d'):
        dot[0] = dot[0]+1
    if key == ord('a'):
        dot[0] = dot[0]-1
    if key == ord('c'): # toggle contour plotting
        if show_contours:
            show_contours = False
        else:
            show_contours = True
    if key == ord('e'): # toggle contour plotting
        if show_ellipse:
            show_ellipse = False
        else:
            show_ellipse = True
    if key == ord('r'): # toggle contour plotting
        if show_rectangle:
            show_rectangle = False
        else:
            show_rectangle = True


cv2.destroyAllWindows()

#print('measured centroid location = ',centroid)
#print('ellipse centre location = ',ellipse[0:2])
#print('manual pupil centre location = ',manual_pup_centre)
print('vector distance from manual to centroid = ',Vcentroid_diff, '(pixels)')
print('vector distance from manual to ellipse = ',Vellipse_diff, '(pixels)')
print('estimated error on these from the manual pupil error is ',vectorUC,'(pixels)')
print(' ')
print('scalar distance from manual to centroid = ',Scentroid_diff, '(pixels)')
print('scalar distance from manual to ellipse = ',Sellipse_diff, '(pixels)')
print('estimated error on these from the manual pupil error is ',scalarUC,'(pixels)')
print('')
print('Normalised Values')
print('vector distance from manual to centroid = ',NVcentroid_diff)
print('vector distance from manual to ellipse = ',NVellipse_diff)
print('estimated error on these from the manual pupil error is ',NvectorUC)
print(' ')
print('Scalar distance from manual to centroid = ',NScentroid_diff)
print('scalar distance from manual to ellipse = ',NSellipse_diff)
print('estimated error on these from the manual pupil error is ',NscalarUC)
