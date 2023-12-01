"Integrantes: Luis Guillermo Flores Ramos, Luis Carlos Quijada Ceceña, Malcom Hiram Navarro López, Jose Benjamin Partida Peralta"

import cv2
import numpy as np

main_image_path = 'C:/Users/memof/Downloads/g-image.png'
template_path = 'C:/Users/memof/Downloads/g-logo.png'

main_image_color = cv2.imread(main_image_path)
if main_image_color is None:
    print("No se pudo cargar la imagen principal.")
    exit(1)

main_image_gray = cv2.cvtColor(main_image_color, cv2.COLOR_BGR2GRAY)

template = cv2.imread(template_path, 0)
if template is None:
    print("No se pudo cargar la imagen de la plantilla.")
    exit(1)

result = cv2.matchTemplate(main_image_gray, template, cv2.TM_CCOEFF_NORMED)

threshold = 0.8
locations = np.where(result >= threshold)
locations = list(zip(*locations[::-1]))  

rectangles = [(*loc, loc[0] + template.shape[1], loc[1] + template.shape[0]) for loc in locations]
scores = [1] * len(rectangles)
pick = cv2.dnn.NMSBoxes(rectangles, scores, threshold, 0.4)

for i in pick.flatten():  
    rect = rectangles[i]
    cv2.rectangle(main_image_color, rect[:2], rect[2:], (0, 255, 0), 2)
    print("Detección:", rect)  

cv2.imshow('Detected', main_image_color)
cv2.waitKey(0)  
cv2.destroyAllWindows()

cv2.imwrite('detected_result.jpg', main_image_color)