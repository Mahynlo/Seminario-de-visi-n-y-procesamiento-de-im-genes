#***************************************************************************************
#Descripcion: Tarea 4 Encontrar líneas usando la transformada probabilística de Hough
#***************************************************************************************

import cv2
import numpy as np

# Load the image
img = cv2.imread('hall-1.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Use HOG descriptor and SVM to detect people in the image
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
rects, weights = hog.detectMultiScale(gray)

# Create a mask where detected people regions are filled with black
mask = np.ones_like(gray) * 255
for (x, y, w, h) in rects:
    mask[y:y+h, x:x+w] = 0

# Detect edges using the Canny detector, but only outside the masked regions
edges = cv2.Canny(cv2.bitwise_and(gray, gray, mask=mask), 100, 150, apertureSize=3)

minLineLength = 10
maxLineGap = 8

# Detect lines
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Save the image with the detected lines
cv2.imwrite('hall-1-HoughLinesP..jpg', img)


