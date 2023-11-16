import cv2
import numpy as np

def salt_and_pepper_noise(image, prob):

    output = np.copy(image)
    # Ruido sal
    num_salt = np.ceil(prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    output[coords[0], coords[1]] = 255

    # Ruido pimienta
    num_pepper = np.ceil(prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    output[coords[0], coords[1]] = 0

    return output

# Leemos la imagen
image = cv2.imread('C:/Users/memof/Downloads/Yandel.jpg', cv2.IMREAD_GRAYSCALE)

# Agregamos ruido con las probabilicades deseadas
image_noise_10 = salt_and_pepper_noise(image, 0.1)
image_noise_25 = salt_and_pepper_noise(image, 0.25)

# Aplicamos los filtros para eliminar el ruido
gaussian_5x5_10 = cv2.GaussianBlur(image_noise_10, (5, 5), 1)
gaussian_11x11_10 = cv2.GaussianBlur(image_noise_10, (11, 11), 3)
median_10 = cv2.medianBlur(image_noise_10, 5)

gaussian_5x5_25 = cv2.GaussianBlur(image_noise_25, (5, 5), 1)
gaussian_11x11_25 = cv2.GaussianBlur(image_noise_25, (11, 11), 3)
median_25 = cv2.medianBlur(image_noise_25, 5)

# Mostramos las imágenes
cv2.imshow('Original', image)
cv2.imshow('Ruido 10%', image_noise_10)
cv2.imshow('Ruido 25%', image_noise_25)
cv2.imshow('Gaussian 5x5 10%', gaussian_5x5_10)
cv2.imshow('Gaussian 11x11 10%', gaussian_11x11_10)
cv2.imshow('Mediana 10%', median_10)
cv2.imshow('Gaussian 5x5 25%', gaussian_5x5_25)
cv2.imshow('Gaussian 11x11 25%', gaussian_11x11_25)
cv2.imshow('Mediana 25%', median_25)

cv2.waitKey(0)
cv2.destroyAllWindows()
