"Integrantes: Luis Guillermo Flores Ramos, Luis Carlos Quijada Ceceña, Malcom Hiram Navarro López, Jose Benjamin Partida Peralta"

import cv2
import os

main_image_path = 'C:/Users/memof/Downloads/coke-bottle.jpg'
template_path = 'C:/Users/memof/Downloads/coke-logo.jpg'

if not os.path.exists(main_image_path) or not os.path.exists(template_path):
    print("Una o más rutas de archivo son incorrectas o los archivos no existen.")
    exit(1)

main_image_color = cv2.imread(main_image_path)  
main_image_gray = cv2.cvtColor(main_image_color, cv2.COLOR_BGR2GRAY)  
template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

if main_image_gray is None or template is None:
    print("No se pudo cargar una o más imágenes.")
    exit(1)

result = cv2.matchTemplate(main_image_gray, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

top_left = max_loc
bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
cv2.rectangle(main_image_color, top_left, bottom_right, (0, 255, 0), 2)

print(f"La mejor coincidencia está en la posición: {top_left}")
print(f"El valor de la mejor coincidencia es: {max_val}")

cv2.imshow('Match', main_image_color)
cv2.imwrite('result.jpg', main_image_color)
cv2.waitKey(0)
cv2.destroyAllWindows()