#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 00:28:47 2020

@author: georgehan
"""
import os
#os.chdir('/Users/georgehan/TDI/Capstone/Smart_Menu')
#os.chdir('/Users/georgehan/GitHub/menu')
print(os.getcwd())
#os.chdir(os.getcwd())
import cv2
import sys

imagePath = "pictures/mcd_very_long.jpg"
cascPath = "haarcascade_pedestrian.xml"
# cascPath = "pedestrian_another.xml"
# cascPath = "haarcascade_fullbody.xml"

pedsCascade =  cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect pedestrian in pic

peds = pedsCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 40)
)

print("Found {0} pedestrian!".format(len(peds)))

# Draw a rectangle around the peds
for (x, y, w, h) in peds:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)

# cv2.imshow("Faces found", image)
status = cv2.imwrite('peds_saved.jpg', image)
print ("Image written to file-system : ",status)
print(imagePath)


#cafe_short: 1.2, 7, 40, 40
#mcd_short: 1.2, 5, 60, 40
#mcd_very_long: 1.2, 5, 60, 40
