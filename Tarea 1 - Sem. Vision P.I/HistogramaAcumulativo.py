#Integrantes: Luis Carlos Quijada Ceceña
#             Luis Guillermo Flores Ramos
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calcular_histograma_acumulativo(imagen):
    histograma = cv2.calcHist([imagen], [0], None, [256], [0,256])
    
    hist_acumulativo = histograma.cumsum()
    
    hist_acumulativo = hist_acumulativo / hist_acumulativo.max()
    
    return hist_acumulativo

if __name__ == "__main__":
    imagen = cv2.imread("C:/Users/luisz/Downloads/dark_street.jpg", cv2.IMREAD_GRAYSCALE)
    
    
    if imagen is None:
        print("Error al cargar la imagen.")
        exit()
    
    hist_acumulativo = calcular_histograma_acumulativo(imagen)
    

    plt.figure(figsize=(6, 4))
    plt.plot(hist_acumulativo, color='black')
    plt.title('Histograma Acumulativo')
    plt.ylabel('H(i)')
    plt.grid(True)
    
    plt.savefig("histograma_acumulativo.png", dpi=300)
    
    print("Imagen del histograma acumulativo guardada en 'histograma_acumulativo.png'.")




