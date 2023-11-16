"Integrantes: Luis Guillermo Flores Ramos, Luis Carlos Quijada Ceceña, Malcom Hiram Navarro López, Jose Benjamin Partida Peralta"

import cv2
import numpy as np

img = cv2.imread('C:/Users/memof/Downloads/Lahore-Fort.jpg', cv2.IMREAD_GRAYSCALE)
img_reference = cv2.imread('C:/Users/memof/Downloads/Lahore-Fort-edges.jpg', cv2.IMREAD_GRAYSCALE)

if img is None or img_reference is None:
    print("Error: Una de las imágenes no ha sido encontrada o tiene un nombre incorrecto. Verifica la ruta.")
    exit()

edges = cv2.Canny(img, 50, 150)

cv2.imwrite("C:/Users/memof/PycharmProject/Tarea3Ejercicio2/Lahore-Fort_Canny.jpg", edges)

img_reference_resized = cv2.resize(img_reference, (edges.shape[1], edges.shape[0]))

difference = cv2.absdiff(img_reference_resized, edges)
squared_difference = np.sum(np.square(difference))
percentage_of_zeros = np.sum(difference == 0) / difference.size * 100

squared_diff = (img_reference_resized.astype(np.float32) - edges.astype(np.float32)) ** 2

squared_diff_normalized = cv2.normalize(squared_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

cv2.imwrite("C:/Users/memof/PycharmProject/Tarea3Ejercicio2/Lahore-Fort_squared_difference.jpg", squared_diff_normalized)

total_difference = np.sum(squared_diff)

zero_percentage = 100 * np.count_nonzero(squared_diff == 0) / squared_diff.size

print(f"Diferencias al cuadrado totales: {total_difference}")
print(f"Porcentaje de ceros (Diferencia al cuadrado): {zero_percentage:.2f}%")
print(f"Diferencia al cuadrado (Comparación directa): {squared_difference}")
print(f"Porcentaje de ceros (Comparación directa): {percentage_of_zeros:.2f}%")

cv2.imshow("Imagen Original", img)
cv2.imshow("Imagen de Referencia", img_reference_resized)
cv2.imshow("Deteccion de bordes con Canny", edges)
cv2.imshow("Diferencia al cuadrado normalizada", squared_diff_normalized)

cv2.waitKey(0)
cv2.destroyAllWindows()