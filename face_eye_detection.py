# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(1)

while 1:
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eye1disp = img*1
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        #if eyes.size != 0:
        if not isinstance(eyes, tuple):
            ex = eyes[:,0]
            ey = eyes[:,1]
            ew = eyes[:,2]
            eh = eyes[:,3]
            i = (np.asarray(np.where(np.amin(ey))))
            i = int(i[0])
            eye1 = img[y+ey[i]:y+ey[i]+eh[i], x+ex[i]:x+ex[i]+ew[i]]

        eye1disp = cv2.resize(eye1,(480,640))
    cv2.imshow('img',eye1disp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#%%
print('eyes.size')
