#Integrantes: Luis Carlos Quijada Ceceña
#             Luis Guillermo Flores Ramos
import cv2
import numpy as np

def convert_to_black_and_white(image_path, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    median_intensity = np.median(img)

    _, thresholded = cv2.threshold(img, median_intensity, 255, cv2.THRESH_BINARY)

    cv2.imwrite(output_path, thresholded)

    cv2.imshow('Black and White Image', thresholded)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_path = 'C:/Users/luisz/Downloads/dark_street.jpg'
output_path = 'C:/Users/luisz/Tarea 1 - Sem. Vision P.I/ImagenModificada.jpg'
convert_to_black_and_white(image_path, output_path)
