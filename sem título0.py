#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 02:49:14 2020

@author: luhan
"""

import cv2
import sys

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)
img_counter = 0

while True:
    #capture frame - by - frame
    ret, frame = video_capture.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k = cv2.waitKey(1)
    faces = faceCascade.detectMultiScale(
        gray, 
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    #draw a rectangle around the faces
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    #Display the result frame
    cv2.imshow('FaceDetection', frame)
    
    if k%256 == 27: # esc key pressed
        break
    elif k%256 == 32: # space pressed
        img_name = 'facedetect_webcam_{}.png'.format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} ESCRITO!".format(img_name))
        img_counter += 1
        
# everything sone
# release the capture
video_capture.release()
cv2.destroyAllWindows()
                                    
