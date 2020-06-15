import matplotlib.pyplot as plt
import numpy as np
import cv2

# need OpenCV version 3.4.7.28
##### press Esc key when running to kill program #####

cap = cv2.VideoCapture(1) # number here chooses which webcam to use (for just one webcam change to 0)
cap.set(3, 1920)
cap.set(4, 1080)

def on_trackbar(val):
    ''' Callback function for changing the threshold value'''
    global thresh # change the global variable for threshold value
    thresh = val
thresh = 35 # initial threshold value
thresh_window = 'Threshold' # put the slider in the threshold window
slider_max = 255 # max slider value
#create trackbar to change threshold
cv2.namedWindow(thresh_window)
thresh_slider = 'Threshold %d' % slider_max
cv2.createTrackbar(thresh_slider, thresh_window , thresh, slider_max, on_trackbar)

def on_trackbar(val2): # slider for zooming on region of interest
    ''' Callback function for changing the size value'''
    global size # change the global variable for size value
    size = val2
size = 80 # initial size value
roi_window = 'Region of Interest' # put the slider in the roi window
slider2_max = 340 # max slider2 value
# create trackbar to change size
cv2.namedWindow(roi_window)
size_slider = 'Size %d' % slider2_max
cv2.createTrackbar(size_slider, roi_window , size, slider2_max, on_trackbar)

def findcentroid(img):
    'For a thresholded img, this ouputs centroid of non zero values'
    indices = np.asarray(np.where(img >0))
    centroid = np.transpose(np.mean(indices,axis=1))
    centroid = np.flip(centroid) #convert to x,y coords with flip
    return centroid

# # creates calibration plot
# object = np.zeros((480,640))
# r = 30
# object[20:20+r, 20:20+r] = 1
# object[430:430+r, 20:20+r] = 1
# object[20:20+r, 590:590+r] = 1
# object[430:430+r, 590:590+r] = 1
#
# # plt.axis('off')
# # fig = plt.imshow(object,"Greys")
# # plt.get_current_fig_manager().full_screen_toggle()
# object = cv2.resize(object,(1280,720))
# # cv2.namedWindow("Calibration", CV_WINDOW_AUTOSIZE)
# # cv2.imshow("Calibration", object)
# # plt.show()


xcall = []
ycall = []
aall = []
ball = []
phiall = []
count = 0
data = -1
wall = []
hall = []
speedall = []

xc = np.nan
yc = np.nan
a = np.nan
b = np.nan
phi = np.nan
speed = np.nan

x = np.nan
y = np.nan
w = np.nan
h = np.nan

yes = 0 # variable for data taking or not, starts in the off state
go = 0

while True:
    ret, frame = cap.read()
    count += 1

    # creates calibration plot
    object = np.zeros((480,640))
    r = 30
    object[20:20+r, 20:20+r] = 1
    object[430:430+r, 20:20+r] = 1
    object[20:20+r, 590:590+r] = 1
    object[430:430+r, 590:590+r] = 1

    # plt.axis('off')
    # fig = plt.imshow(object,"Greys")
    # plt.get_current_fig_manager().full_screen_toggle()
    # object = cv2.resize(object,(1280,720))
    # cv2.namedWindow("Calibration", CV_WINDOW_AUTOSIZE)
    # cv2.imshow("Calibration", object)
    # plt.show()

    ellipse = ((np.nan,np.nan),(np.nan,np.nan),np.nan)
    k, l, _ = frame.shape

    centerk = int(k/2)
    scalek = int(k/12)
    centerl = int(l/2)
    scalel = int(l/12)

    roi = frame[centerk-int(size*0.75):centerk+int(size*0.75),centerl-size:centerl+size]

    rows, cols, _ = roi.shape

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    _, threshold = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        if cnt.shape[0]>=5:
            # (x, y, w, h) = cv2.boundingRect(cnt)
            pupil = cv2.drawContours(roi, [cnt], -1, (0, 255, 255), 1)
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(roi, ellipse, (255,0, 255), 1, cv2.LINE_AA)

        # cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 1)
        cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 1)

        break

    sizing = size/slider2_max
    ratiol = ((cols)/(1280))#*(size)
    ratiok = ((rows)/(720))#*(size*0.75)
    object = cv2.resize(object,(int(1280*sizing),int(720*sizing)))
    # print(ratiok)

    if np.sum(threshold)>0:    # only if there are non zero values in threshold image
        centroid = findcentroid(threshold)
        cent_coord = centroid#.astype(int) #convert to integer for plotting
        roi[int(cent_coord[1]),int(cent_coord[0])]=(0,0,255) # plot red pixel at centre of eye
        object[int((cent_coord[1]/ratiok)*sizing)-5:int((cent_coord[1]/ratiok)*sizing)+5,int((cent_coord[0]/ratiol)*sizing)-5:int((cent_coord[0]/ratiol)*sizing)+5]= 200

    [xc,yc] = ellipse[0]
    [a,b] = ellipse[1]
    phi = ellipse[2]

    key = cv2.waitKey(1) # checks for keyboard presses

    if key == ord("q"): # checks for 'q' pressed to start taking data
        yes = 1
        data += 1
        go = 1
        # if data >= 1:
        #     xcall[:,np.newaxis]
        #     ycall[:,np.newaxis]
        #     aall[:,np.newaxis]
        #     ball[:,np.newaxis]
        #     phiall[:,np.newaxis]
        #     break
    # print(data)
    if key == ord("p"): # checks for 'p' pressed to stop taking data
        yes = 0

    if yes == 1: # appends data if 'q' pressed
        xcall.append(xc)
        ycall.append(yc)
        aall.append(a)
        ball.append(b)
        phiall.append(phi)
        wall.append((x+(w/2)))
        hall.append((y+(h/2)))
        speedall.append(count)
    else: # appends nans if data is not being taken
        xcall.append(np.nan)
        ycall.append(np.nan)
        aall.append(np.nan)
        ball.append(np.nan)
        phiall.append(np.nan)
        wall.append((np.nan))
        hall.append((np.nan))
        speedall.append((np.nan))


    colresize = round(l/2.5)
    rowrresize = round(k/2.5)
    resized = (colresize,rowrresize)
    roi = cv2.resize(roi, resized)
    # roi = cv2.resize(roi, (1920,1080))
    # roi = cv2.resizeWindow(1920,1080);
    threshold = cv2.resize(threshold, resized)
    roi = np.flip(roi,axis=1)
    threshold = np.flip(threshold,axis=1)
    frame = np.flip(frame,axis=1)
    object = np.flip(object,axis=1)
    cv2.imshow("Threshold", threshold)
    cv2.imshow("Calibration", object)
    # cv2.imshow("Original",frame)
    cv2.imshow("Region of Interest", roi)

    if key == 27:
        break

min = np.nanmin(speedall)
max = np.nanmax(speedall)
speeddiff = max - min
distsq = (np.nanmax(xcall)-np.nanmin(xcall))**2 + (np.nanmax(ycall)-np.nanmin(ycall))**2
dist = np.sqrt(distsq)
fast = speeddiff/dist
print(fast)

time = np.linspace(1,count,count)

# print(xcall)
# xcall[xcall] != np.nan
# print(B)

# xdiff = abs(xcall-wall)
# ydiff = abs(ycall-hall)
if go == 1:
    plt.title('Ellipse Centroid Coordinates')
    plt.xlabel('xc')
    plt.ylabel('yc')
    plt.xlim(cols,0)
    plt.ylim(rows,0)
    plt.grid('on')
    # plt.gca().invert_xaxis()
    # for i in range(0,data):
    plt.plot(xcall,ycall,'-k.',label = 'Ellipse Centre')


    # plt.figure()
    # plt.title('Contour Center Coordinates')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.xlim(cols,0)
    # plt.ylim(rows,0)
    # plt.grid('on')
    # plt.gca().invert_xaxis()
    plt.plot(wall,hall,'-g.',label = 'Contour Centre')
    plt.legend()

    # plt.figure()
    # plt.title('Coordinate Difference')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.xlim(cols,0)
    # plt.ylim(rows,0)
    # plt.grid('on')
    # # plt.gca().invert_xaxis()
    # plt.plot(xdiff,ydiff,'-b.')

    plt.figure()
    plt.title('Ellipse Angle Over Time')
    plt.xlabel('Time (frames)')
    plt.ylabel('Phi (degrees)')
    plt.plot(time,phiall,'-ro')

    plt.show()

cap.release()
cv2.destroyAllWindows()
