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
trainedWeights = "trained_customer.xml"

customerTrainedWeights =  cv2.CascadeClassifier(trainedWeights)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect customer in pic

customers = customerTrainedWeights.detectMultiScale(
        gray,
        scaleFactor=1.2, # controlls fine (smaller) vs coarse (larger) trade off, needs to > 1.0
        minNeighbors=5, # used to combine overlapping small boxes into big one
        minSize=(60, 40) # box size, distance between recoginized customers
)

print("Found {0} customers!".format(len(customers)))

# Draw a rectangle around the customers
for (x, y, w, h) in customers:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# cv2.imshow("customers found", image)
status = cv2.imwrite('customers_detect.jpg', image)
print ("Image written to file-system : ",status)
print(imagePath)

# hyperparameter sets:
# cafe_short: 1.2, 7, 40, 40
# mcd_short: 1.2, 5, 60, 40
# mcd_very_long: 1.2, 5, 60, 40
