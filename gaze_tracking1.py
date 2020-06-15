# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:41:03 2019

@author: ppyhb1
"""
#%% import libraries and define functions
# need OpenCV version 3.4.7.28
#pip install opencv-python==3.4.7.28
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

def on_trackbar(val):
    ''' Callback function for changing the threshold value'''
    global thresh # change the global variable for threshold value
    thresh = val

def threshold(img):
    '''Function to threshold eye image using opening and bluring'''
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)  #blur before thresholding to remove noise
    _, imgthreshold = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY_INV)
    #imgthresholdold = 1*imgthreshold # duplicate the one before opening for comparison
    imgthreshold = cv2.morphologyEx(imgthreshold, cv2.MORPH_OPEN, open_kernel)
    #imgthreshold = cv2.medianBlur(imgthreshold, 5)
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
    x1 = np.amin(x)
    x2 = np.amax(x)
    y1 = np.amin(y)
    y2 = np.amax(y)
#    xrange = np.ptp(x) # range of x and y
#    yrange = np.ptp(y)
#    x1 = np.mean([z for z in x if z < xrange/2])
#    x2 = np.mean([z for z in x if z > xrange/2])
#    y1 = np.mean([z for z in x if z < yrange/2])
#    y2 = np.mean([z for z in x if z > yrange/2])
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


#%%
# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier(r'C:\Users\jason\OneDrive - The University of Nottingham\4th Year\Image Processing Files\Mini Project\haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier(r'C:\Users\jason\OneDrive - The University of Nottingham\4th Year\Image Processing Files\Mini Project\haarcascade_eye.xml')

cap = cv2.VideoCapture(1)
# make 720p
#cap.set(3,1280)
#cap.set(4,720)
# make 480p
#cap.set(3,640)
#cap.set(4,480)
# make 1080p
#cap.set(3,1920)
#cap.set(4,1080)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
original_frames_out = cv2.VideoWriter('original_frames.avi',fourcc, 10.0, (640,480))
frames_out = cv2.VideoWriter('frames.avi',fourcc, 10.0, (640,480))

thresh = 60# initial threshold value

thresh_window = 'Eyes Threshold' # put the slider in the threshold window
slider_max = 255 # max slider value
#create trackbar to change threshold
cv2.namedWindow(thresh_window)
thresh_slider = 'thresh %d' % slider_max
cv2.createTrackbar(thresh_slider, thresh_window , thresh, slider_max, on_trackbar)

callib_img_size = (1280,720) # size of the callibration image
callibrated = True # Boolean to tell if callibrated yet

start_time = time.time() # time at start of video capture
frame_time = 1 # number of seconds for stationary frame to diplay eyes
frame_start_time = time.time() - frame_time -1 #intitial time for frame timer
# set up array to store timestamps in
timestamps = np.array([]).reshape(0,1)
#set up array for face box parameters to be stored in
Faces = np.array([]).reshape(0,4)
#set up array for eye box parameters to be stored in
Leyes = np.array([]).reshape(0,4)
Reyes = np.array([]).reshape(0,4)
#set up array for pupil centroid coords to be stored in
Lcentroid = np.array([]).reshape(0,2)
Rcentroid = np.array([]).reshape(0,2)
#Set up array for ellipse parameters to be stored
Lellipse = np.array([]).reshape(0,5)
Rellipse = np.array([]).reshape(0,5)

Lcalib_rect_corners = np.array([np.nan,np.nan,np.nan,np.nan])
Rcalib_rect_corners = np.array([np.nan,np.nan,np.nan,np.nan])

#get the first frame to get frame size
_, frame = cap.read()
# set up array to store recorded frames
#Frames = np.zeros((frame.shape[0],frame.shape[1],3))
#Frames_original = np.zeros((frame.shape[0],frame.shape[1],3))
#Frames = np.array([]).reshape(frame.shape[0],frame.shape[1],3,0)
#Frames_original = np.array([]).reshape(frame.shape[0],frame.shape[1],3,0)

#set up dictionaries for storing recording sessions
FacesAll = {}
LeyesAll = {}
ReyesAll = {}
LcentroidAll = {}
RcentroidAll = {}
LellipseAll = {}
RellipseAll = {}
#FramesAll = {}
#Frames_originalAll = {}



open_kernel = np.ones((4,4)) #kernal for opening to remove non pupil parts of eye

recording = False #bollean for recording data
prevkey = [] #define previous key value
record_count = 0 #define count for number of recordings taken
num_of_cal_points = 4 #number of callibration points to look at

#Set up array for callibration pupil location points
Lcentroid_calib_mean = np.zeros((num_of_cal_points,2))
Lcentroid_calib_var = np.zeros((num_of_cal_points,2))
Rcentroid_calib_mean = np.zeros((num_of_cal_points,2))
Rcentroid_calib_var = np.zeros((num_of_cal_points,2))

lgaze = np.array([np.nan,np.nan]) # set gaze coords to nan
rgaze = np.array([np.nan,np.nan])



while True:
    timestamps = np.vstack((timestamps,(time.time()-start_time))) #record time
    ret, frame = cap.read()
    frame_original = frame*1 # copy the original frame
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
    leye = frame*0 + 255 # so that leye can be defined if no eyes are found
    reye = frame*0 + 255 # 255 so that pupil centroid not found if no eyes found
    #leye = np.ones((frame.shape[0],frame.shape[1],frame.shape[2]))*255  # so that leye can be defined if no eyes are found
    #reye = np.ones((frame.shape[0],frame.shape[1],frame.shape[2]))*255
    lellipse = ((np.nan,np.nan),(np.nan,np.nan),np.nan) #so nan ellipse can fill
    rellipse = ((np.nan,np.nan),(np.nan,np.nan),np.nan) #array when no ellipse drawn
    lcentroid = np.array([np.nan,np.nan])
    rcentroid = np.array([np.nan,np.nan])

    if not isinstance(faces, tuple): # if a face is present / prevents err when no face
        fx,fy,fw,fh = faces[0,:] # select first face to use
        cv2.rectangle(frame,(fx,fy),(fx+fw,fy+fh),(255,0,0),2)
        face_gray = frame_gray[fy:fy+fh, fx:fx+fw]
        face_color = frame[fy:fy+fh, fx:fx+fw]

        # only look for eyes in top half of face
        # look for left eye
        eyes = eye_cascade.detectMultiScale(face_gray[0:int(fh/2),:])

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(face_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        if not isinstance(eyes, tuple): # prevents error when no eyes found
            ex = eyes[:,0]
            ey = eyes[:,1]
            ew = eyes[:,2]
            eh = eyes[:,3]
            # select left and right eye box locations
            for a in range(0,eyes.shape[0]):
                if ex[a] > fw/2: # if eye on left side of face
                    #coords of left eye box paramaters
                    leyes = eyes[a,:]*1 # *1 to copy instead of duplicate
                    leye = frame[fy+ey[a]:fy+ey[a]+eh[a], fx+ex[a]:fx+ex[a]+ew[a]]
                if ex[a] < fw*2/5: # if eye on right of face
                    reyes = eyes[a,:]*1 # *1 to copy instead of duplicate
                    reye = frame[fy+ey[a]:fy+ey[a]+eh[a], fx+ex[a]:fx+ex[a]+ew[a]]
            #li = (np.asarray(np.where(ex > fw/2))) # index for left eye
            #li = int(li[0,0]) # only take scalar from size 1 array
            #ri = (np.asarray(np.where(ex < fw/2))) # index for right eye
            #ri = int(ri[0,0])
#            li = (np.asarray(np.where(ex == np.amax(ex)))) # index for left eye
#            li = int(li[0,0]) # only take scalar from size 1 array
#            ri = (np.asarray(np.where(ex == np.amin(ex)))) # index for right eye
#            ri = int(ri[0,0])
#            leye = frame[fy+ey[li]:fy+ey[li]+eh[li], fx+ex[li]:fx+ex[li]+ew[li]]
#            reye = frame[fy+ey[ri]:fy+ey[ri]+eh[ri], fx+ex[ri]:fx+ex[ri]+ew[ri]]
#            leyes = eyes[li,:]*1 # get x,y,w,h of eye boxes for both eyes
#            reyes = eyes[ri,:]*1 # *1 to stop eyes being changed by leyes being changed
            if eyes.shape[0] == 1: # if only one eye found
                if ex > fw*2/5: # if eye on left side of face
                    reyes = np.empty(4)
                    reyes[:] = np.nan
                else:
                    leyes = np.empty(4)
                    leyes[:] = np.nan
        else:   # else if no eyes are found, set eyes as nan
            eyes = np.empty((2,4))
            eyes[:] = np.nan
            leyes = eyes[0,:]*1 # set x,y,w,h, for both eyes to nan
            reyes = eyes[0,:]*1
    else:
        faces = np.empty((1,4)) #set faces to nan if no face (1,4) is needed
        faces[:] = np.nan
        eyes = np.empty((2,4)) # set eyes to nan if no face
        eyes[:] = np.nan
        leyes = eyes[0,:]*1 # set x,y,w,h, for both eyes to nan
        reyes = eyes[0,:]*1

    #find contour and centroid of pupil of each eye, fit ellipse to this
    #Left eye
    lthreshold, lcontours, leye_gray = threshold(leye) #threshold eye
    # Find centroid of left pupil
    if np.sum(lthreshold)>0:    # only if there are non zero values in threshold image
        lcentroid = findcentroid(lthreshold)
        lcent_coord = lcentroid.astype(int) #convert to integer for plotting
        leye[lcent_coord[1],lcent_coord[0]]=(0,0,255) # plot red pixel at centre of eye
        #Alternatively plot circle at centre of pupil
        #cv2.circle(leye, (lcent_coord[1],lcent_coord[0]), 1, (0,0,255), thickness=1, lineType=8, shift=0)

    #draw largest contour of left pupil
    '''for cnt in lcontours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        lpupil = cv2.drawContours(leye, [cnt], -1, (0, 255, 255), 1)
        if cnt.shape[0]>=5: #if at least 5 points on contour
            lellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(leye, lellipse, (255,0, 255), 1, cv2.LINE_AA)
        #Optionally plot rectangle and lines around contour
        #cv2.rectangle(leye, (x, y), (x + w, y + h), (0, 0, 255), 1)
        #cv2.line(leye, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 1)
        #cv2.line(leye, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 1)
        break #to only plot largest contour'''

    #Convert ellipse parameters to an array for storing
    lellipse=np.hstack([lellipse[0],lellipse[1],lellipse[2]])



    # Right eye
    rthreshold, rcontours, reye_gray = threshold(reye) #threshold eye
    # Find centroid of left pupil
    if np.sum(rthreshold)>0:    # only if there are non zero values in threshold image
        rcentroid = findcentroid(rthreshold)
        rcent_coord = rcentroid.astype(int) #convert to integer for plotting
        reye[rcent_coord[1],rcent_coord[0]]=(0,0,255) # plot red pixel at centre of eye
        #Alternatively plot circle at centre of pupil
        #cv2.circle(leye, (lcent_coord[1],lcent_coord[0]), 1, (0,0,255), thickness=1, lineType=8, shift=0)

    #draw contours of right pupil
    '''for cnt in rcontours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        rpupil = cv2.drawContours(reye, [cnt], -1, (0, 255, 255), 1)
        if cnt.shape[0]>=5: #if at least 5 points on contour
            rellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(reye, rellipse, (255,0, 255), 1, cv2.LINE_AA)
        #Optionally plot rectangle and lines around contour
        #cv2.rectangle(reye, (x, y), (x + w, y + h), (0, 0, 255), 1)
        #cv2.line(reye, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 1)
        #cv2.line(reye, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 1)
        break #to only plot largest contour'''
    #Convert ellipse parameters to an array and store for each vid frame
    rellipse=np.hstack([rellipse[0],rellipse[1],rellipse[2]])


    # Convert coordinates so they are all relative to camera
    for a in range (0,eyes.shape[0]):
        eyes[a,0:2] = faces[0,0:2] + eyes[a,0:2] #needed to define eye frame
    leyes[0:2] = faces[0,0:2] + leyes[0:2]
    reyes[0:2] = faces[0,0:2] + reyes[0:2]
    # pupil centroid coords, (from threshold centroid)
    lcentroid = leyes[0:2] + lcentroid
    rcentroid = reyes[0:2] + rcentroid
    #pupil ellipse centres
    lellipse[0:2] = leyes[0:2] + lellipse[0:2]
    rellipse[0:2] = reyes[0:2] + rellipse[0:2]

    #Generate callibration image
    callib_image = np.zeros((480,640,3))
    p = 15
    pips = np.array([[p,p],[callib_image.shape[1]-p,p],[callib_image.shape[1]-p,callib_image.shape[0]-p],[p,callib_image.shape[0]-p]])
    for a in range(0,pips.shape[0]):
        x,y = pips[a,:]
        cv2.circle(callib_image, (x,y), 5, (255,255,255), thickness=5, lineType=8, shift=0)


    # Once callibrated, find the gaze positions of each eye
    if callibrated:
        lgaze = findgaze(lcentroid, Lcalib_rect_corners, callib_image) # rect corners are defined after calibration
        rgaze = findgaze(rcentroid, Rcalib_rect_corners, callib_image)
        #lgaze = findgaze(lcentroid, Lcalib_rect_corners) # rect corners are defined after calibration
        #rgaze = findgaze(rcentroid, Rcalib_rect_corners)


    #plot location of gaze for each eye on callib image
    #cv2.circle(callib_image, lcentroid.astype(int), 2, (255,0,0), thickness=2, lineType=8, shift=0)
    #cv2.circle(callib_image, rcentroid.astype(int), 2, (0,0,255), thickness=2, lineType=8, shift=0)
    if not np.isnan(lgaze).any():
        cv2.circle(callib_image, (int(lgaze[0]),int(lgaze[1])), 2, (255,0,0), thickness=2, lineType=8, shift=0)
    if not np.isnan(rgaze).any():
        cv2.circle(callib_image, (int(rgaze[0]),int(rgaze[1])), 2, (0,0,255), thickness=2, lineType=8, shift=0)
    cv2.imshow('callibration Image',cv2.resize(np.flip(callib_image,axis=1),callib_img_size))



    # thresholded whole image for manual threshold setting display
    frame_threshold, _, _ = threshold(frame_original)

    # Display full frame and full thresholded frame
    cv2.imshow("Frame", np.flip(cv2.resize(frame, (1280,720)),axis=1))
    cv2.imshow("Frame Threshold", cv2.resize(np.flip(frame_threshold,axis=1),(1280,720)))

    # Displaying only the eyes next to eachother
    '''leyedisp = cv2.resize(leye,(480,640))
    reyedisp = cv2.resize(reye,(480,640))
    eyesdisp = np.hstack([reyedisp,leyedisp])
    cv2.imshow('Eyes',np.flip(eyesdisp,axis=1))'''

    #Display thresholded eyes next to eachother
    leyedisp_thresh = cv2.resize(lthreshold,(480,640))
    reyedisp_thresh = cv2.resize(rthreshold,(480,640))
    eyesdisp_thresh = np.hstack([reyedisp_thresh,leyedisp_thresh])
    cv2.imshow('Eyes Threshold',np.flip(eyesdisp_thresh,axis=1))

    #coordinates of the frame showing only the eye region, these coordinates update
    # every frame_time seconds to give a new frame for the eye region
    # result is to display the eyes in a frame that is stationary relative to
    # camera for frame_time seconds
    if time.time()-frame_start_time >= frame_time:
        frame_start_time = time.time()
        if (np.isnan(eyes)).any(): # This is to prevent error when no face/eyes are found
            efy1 = 0
            efy2 = 200
            efx1 = 0
            efx2 = 200
        else:
            border = 10
            efy1 = eyes[0,1] - border
            efy2 = eyes[0,1] + eyes[0,3] + border
            efx1 = fx
            efx2 = fx + fw
            #efy1 = reyes[1] - border
            #efy2 = reyes[1] + reyes[3] + border
            #efx1 = reyes[0] - border
            #efx2 = leyes[0]+ reyes[2] + border

    eye_frame = frame[efy1:efy2,efx1:efx2]
    eyes_threshold = frame_threshold[efy1:efy2,efx1:efx2]
    scale = 4
    #cv2.imshow('Eye Frame',np.flip(eye_frame,axis=1))
    #cv2.imshow('Eyes Threshold',np.flip(eyes_threshold,axis=1))
    cv2.imshow('Eye Frame',cv2.resize(np.flip(eye_frame,axis=1),(eye_frame.shape[1]*scale,eye_frame.shape[0]*scale)))
    cv2.imshow('Eyes Threshold',cv2.resize(np.flip(eyes_threshold,axis=1),(eyes_threshold.shape[1]*scale,eyes_threshold.shape[0]*scale)))


    key = cv2.waitKey(1)
    if key == 27: # press Esc to quit
        break

    #Append data to arrays to store data
    if key == ord('r') and prevkey != ord('r'): #if r key has just been pressed
        # change the recording boolean
        if recording:
            recording = False
            # Add data to tuple element for that recordinf session and reset
            # recording array

            FacesAll['recording_' + str(record_count)] = Faces
            LeyesAll['recording_' + str(record_count)] = Leyes
            ReyesAll['recording_' + str(record_count)] = Reyes
            LcentroidAll['recording_' + str(record_count)] = Lcentroid
            RcentroidAll['recording_' + str(record_count)] = Rcentroid
            LellipseAll['recording_' + str(record_count)] = Lellipse
            RellipseAll['recording_' + str(record_count)] = Rellipse


            #reset variables
            Faces = np.array([]).reshape(0,4)
            #set up array for eye box parameters to be stored in
            Leyes = np.array([]).reshape(0,4)
            Reyes = np.array([]).reshape(0,4)
            #set up array for pupil centroid coords to be stored in
            Lcentroid = np.array([]).reshape(0,2)
            Rcentroid = np.array([]).reshape(0,2)
            #Set up array for ellipse parameters to be stored
            Lellipse = np.array([]).reshape(0,5)
            Rellipse = np.array([]).reshape(0,5)

            if record_count == num_of_cal_points: # Once callibrated
                for a in range(0,num_of_cal_points):
                    #find average of each recording position
                    Lcentroid_pos = LcentroidAll['recording_' + str(a+1)] #a+1 because recording start at 1
                    Lcentroid_calib_mean[a,:] = np.nanmean(Lcentroid_pos,axis=0)
                    Lcentroid_calib_var[a,:] = np.nanvar(Lcentroid_pos,axis=0)

                    Rcentroid_pos = RcentroidAll['recording_' + str(a+1)]
                    Rcentroid_calib_mean[a,:] = np.nanmean(Rcentroid_pos,axis=0)
                    Rcentroid_calib_var[a,:] = np.nanvar(Rcentroid_pos,axis=0)
                    # np.nanmean needs to be used to ignore nan values

                # Fit a rectangle to callibration points
                Lcalib_rect, Lcalib_rect_corners = fitrectangle(Lcentroid_calib_mean)
                Rcalib_rect, Rcalib_rect_corners = fitrectangle(Rcentroid_calib_mean)

                callibrated = True # change callibration boolean to true

        else:
            recording = True
            record_count = record_count + 1

    if recording:
        # Append face and eye box parameters to array
        Faces = np.vstack([Faces,faces])
        Leyes = np.vstack([Leyes,leyes])
        Reyes = np.vstack([Reyes,reyes])

        # write the video frame
        frames_out.write(frame)
        original_frames_out.write(frame_original)

        Lellipse = np.vstack([Lellipse,lellipse])
        # append left pupil centroid coordinate
        Lcentroid = np.vstack((Lcentroid,lcentroid))

        Rellipse = np.vstack([Rellipse,rellipse])
        # append left pupil centroid coordinate
        Rcentroid = np.vstack((Rcentroid,rcentroid))

    keyprev = key # update previous value of key to test for key change


cap.release()
frames_out.release()
original_frames_out.release()
cv2.destroyAllWindows()


''' Axes need inverting'''
#Plot pupil stuff
plt.figure()
plt.title('Left Centroid Coordinates')
plt.xlabel('x')
plt.ylabel('y')
#plt.xlim(cols,0)
#plt.ylim(rows,0)
plt.grid('on')
#plt.gca().invert_xaxis()
plt.plot(Lcentroid_calib_mean[:,0],Lcentroid_calib_mean[:,1],'o')
plt.plot(Lcentroid_calib_mean[:,0],Lcentroid_calib_mean[:,1],'o')
plt.plot(Lcalib_rect[:,0],Lcalib_rect[:,1],'-b')

plt.figure()
ax = plt.axes()
plt.title('Left Centroid Coordinates')
plt.xlabel('x')
plt.ylabel('y')
ax.grid('on')
for a in range(0,record_count):
    #find average of each recording position
    Lcentroid_pos = LcentroidAll['recording_' + str(a+1)] #a+1 because recording start at 1
    ax.plot(Lcentroid_pos[:,0],Lcentroid_pos[:,1],'.',label = ('recording' + str(a+1)))
    ax.legend()

# Right eye
plt.figure()
plt.title('Right Centroid Coordinates')
plt.xlabel('x')
plt.ylabel('y')
#plt.xlim(cols,0)
#plt.ylim(rows,0)
plt.grid('on')
#plt.gca().invert_xaxis()
plt.plot(Rcentroid_calib_mean[:,0],Rcentroid_calib_mean[:,1],'o')
plt.plot(Rcentroid_calib_mean[:,0],Rcentroid_calib_mean[:,1],'o')
plt.plot(Rcalib_rect[:,0],Rcalib_rect[:,1],'-b')

plt.figure()
ax = plt.axes()
plt.title('Right Centroid Coordinates')
plt.xlabel('x')
plt.ylabel('y')
ax.grid('on')
for a in range(0,record_count):
    #find average of each recording position
    Rcentroid_pos = RcentroidAll['recording_' + str(a+1)] #a+1 because recording start at 1
    ax.plot(Rcentroid_pos[:,0],Rcentroid_pos[:,1],'.',label = ('recording' + str(a+1)))
    ax.legend()

#plt.plot(LellipseAll['recording_1'][:,0],LellipseAll['recording_1'][:,1],'-b.',label = 'Ellipse Centre')
#plt.plot(RellipseAll['recording_1'][:,0],RellipseAll['recording_1'][:,1],'-r.',label = 'Ellipse Centre')
#plt.plot(FacesAll['recording_1'][:,0],FacesAll['recording_1'][:,1],'-k')
#plt.plot(LeyesAll['recording_1'][:,0],LeyesAll['recording_1'][:,1],'-g')
plt.show()
