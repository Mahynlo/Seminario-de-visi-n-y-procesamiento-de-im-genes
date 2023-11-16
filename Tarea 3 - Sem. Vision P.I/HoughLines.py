#*******************************************************************************
#Descripcion: Tarea 4 Encontrar líneas usando la transformada standard de Hough
#*******************************************************************************

import cv2
import numpy as np

# Load the image
img = cv2.imread('hall-1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Use HOG descriptor and SVM to detect people in the image
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
rects, weights = hog.detectMultiScale(gray, winStride=(8, 8), padding=(16, 16))

# Create a mask of the same size as the image and fill it with white (255)
mask = np.ones_like(gray) * 255
for (x, y, w, h) in rects:
    # Fill the detected human regions with black (0) in the mask
    mask[y:y+h, x:x+w] = 0

# Detect edges using the Canny detector on regions not masked out
masked_edges = cv2.bitwise_and(gray, gray, mask=mask)
edges = cv2.Canny(masked_edges, 350, 300, apertureSize=3)

lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

# Iterate over the detected lines
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 2000*(-b))
    y1 = int(y0 + 2000*(a))
    x2 = int(x0 - 2000*(-b))
    y2 = int(y0 - 2000*(a))

    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Save the resulting image
cv2.imwrite('hall-1-HoughLines.jpg', img)


