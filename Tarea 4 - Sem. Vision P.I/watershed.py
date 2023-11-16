"Integrantes: Luis Guillermo Flores Ramos, Luis Carlos Quijada Ceceña, Malcom Hiram Navarro López, Jose Benjamin Partida Peralta"

import cv2
import numpy as np

def segment_coins(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.medianBlur(gray, 25)

    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)

    markers = markers + 1
    markers[unknown == 255] = 0

    cv2.watershed(img, markers)
    img[markers == -1] = [0, 0, 255]

    cv2.imshow('Monedas Detectadas', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Aplicar para las dos imágenes
image_paths = ['C:/Users/luisz/Downloads/coins1.png', 'C:/Users/luisz/Downloads/coins2.png']
for path in image_paths:
    segment_coins(path)
