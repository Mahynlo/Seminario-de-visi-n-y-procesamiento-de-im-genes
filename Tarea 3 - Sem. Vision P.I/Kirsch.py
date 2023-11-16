"Integrantes: Luis Guillermo Flores Ramos, Luis Carlos Quijada Ceceña, Malcom Hiram Navarro López, Jose Benjamin Partida Peralta"

import cv2
import numpy as np

kernels = [
    np.array([[-3, -3,  5],
              [-3,  0,  5],
              [-3, -3,  5]]),
    
    np.array([[-3,  5,  5],
              [-3,  0,  5],
              [-3, -3, -3]]),

    np.array([[ 5,  5,  5],
              [-3,  0, -3],
              [-3, -3, -3]]),

    np.array([[ 5,  5, -3],
              [ 5,  0, -3],
              [-3, -3, -3]]),

    np.array([[ 5, -3, -3],
              [ 5,  0, -3],
              [ 5, -3, -3]]),

    np.array([[-3, -3, -3],
              [ 5,  0, -3],
              [ 5,  5, -3]]),

    np.array([[-3, -3, -3],
              [-3,  0, -3],
              [ 5,  5,  5]]),

    np.array([[-3, -3,  5],
              [-3,  0,  5],
              [-3,  5,  5]])
]

image = cv2.imread('C:/Users/luisz/Downloads/Lahore-Fort.jpg', cv2.IMREAD_GRAYSCALE)

outputs = []
for kernel in kernels:
    output = cv2.filter2D(image, cv2.CV_32F, kernel)
    outputs.append(output)

magnitude = np.sqrt(sum(np.square(output) for output in outputs))

normalized_magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

_, thresholded_magnitude = cv2.threshold(normalized_magnitude, 50, 255, cv2.THRESH_BINARY)

reference_image = cv2.imread('C:/Users/luisz/Downloads/Lahore-Fort-edges.jpg', cv2.IMREAD_GRAYSCALE)

magnitude_resized = cv2.resize(normalized_magnitude, (reference_image.shape[1], reference_image.shape[0]))

difference = cv2.absdiff(reference_image, magnitude_resized)
squared_difference = np.sum(np.square(difference))
percentage_of_zeros = np.sum(difference == 0) / difference.size * 100

print(f"Diferencia al cuadrado: {squared_difference}")
print(f"Porcentaje de ceros: {percentage_of_zeros:.2f}%")

cv2.imwrite('C:/Users/luisz/Downloads/Kirsch_Result_Normalized.jpg', normalized_magnitude)

cv2.imshow('Imagen Original', image)
cv2.imshow('Kirsch', normalized_magnitude)
cv2.imshow('Imagen Comparativa', reference_image)

cv2.waitKey(0)
cv2.destroyAllWindows()