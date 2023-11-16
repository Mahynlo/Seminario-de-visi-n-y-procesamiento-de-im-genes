import cv2
import numpy as np

def sharpen_image(image, c):
    
    kernel = np.array([
        [0, c/4, 0],
        [c/4, 1-c, c/4],
        [0, c/4, 0]
    ])
    
    sharpened = cv2.filter2D(image, -1, kernel)
    
    return sharpened

image_path = 'C:/Users/luisz/Downloads/Paisaje.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

c_value = 2.0  
sharpened = sharpen_image(image, c_value)

cv2.imshow('Original', image)
cv2.imshow('Sharpened', sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()
