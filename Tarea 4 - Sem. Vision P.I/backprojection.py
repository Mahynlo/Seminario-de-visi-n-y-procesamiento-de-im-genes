"Integrantes: Luis Guillermo Flores Ramos, Luis Carlos Quijada Ceceña, Malcom Hiram Navarro López, Jose Benjamin Partida Peralta"

import cv2
import numpy as np

def remove_ball(image_path, roi_path):

    img = cv2.imread(image_path)
    roi = cv2.imread(roi_path)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Backprojection
    dst = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)

    cv2.normalize(dst, dst, 0, 255, cv2.NORM_MINMAX)
    dst = cv2.GaussianBlur(dst, (5,5), 0)
    _, mask = cv2.threshold(dst, 15, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.erode(mask, None, iterations=3)
    mask = cv2.dilate(mask, None, iterations=3)
    
    result = cv2.bitwise_and(img, img, mask=255-mask)
    
    cv2.imshow('Imagen Sin Balon', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

remove_ball("C:/Users/luisz/Downloads/fussball-orange.jpg", "C:/Users/luisz/Downloads/Pelota.png")



