#Integrantes: Luis Carlos Quijada Ceceña
#             Luis Guillermo Flores Ramos

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

OUTPUT_PATH = 'C:/Users/luisz/Tarea 1 - Sem. Vision P.I'

def save_histogram_as_image(image, filename):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    
    plt.figure()
    plt.bar(range(256), hist.ravel(), width=1, color='b')
    plt.title('Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    plt.savefig(os.path.join(OUTPUT_PATH, filename))
    plt.close()

def histogram_equalization(image):
    return cv2.equalizeHist(image)

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

def calculate_mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def process_image(image_name):
    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)

    save_histogram_as_image(image, os.path.basename(image_name).split('.')[0] + "_original_histogram.png")

    equalized_image = histogram_equalization(image)
    cv2.imwrite(os.path.join(OUTPUT_PATH, os.path.basename(image_name).split('.')[0] + "_equalized.png"), equalized_image)
    save_histogram_as_image(equalized_image, os.path.basename(image_name).split('.')[0] + "_equalized_histogram.png")

    clahe_image = apply_clahe(image)
    cv2.imwrite(os.path.join(OUTPUT_PATH, os.path.basename(image_name).split('.')[0] + "_clahe.png"), clahe_image)
    save_histogram_as_image(clahe_image, os.path.basename(image_name).split('.')[0] + "_clahe_histogram.png")

    
    mse_equalized = calculate_mse(image, equalized_image)
    mse_clahe = calculate_mse(image, clahe_image)

    print(f"MSE between original and equalized for {image_name}: {mse_equalized}")
    print(f"MSE between original and CLAHE for {image_name}: {mse_clahe}")

    cv2.imshow('Original', image)
    cv2.imshow('Equalized', equalized_image)
    cv2.imshow('CLAHE', clahe_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

process_image('C:/Users/luisz/Downloads/checkerboard.png')
process_image('C:/Users/luisz/Downloads/dark_street.jpg')
