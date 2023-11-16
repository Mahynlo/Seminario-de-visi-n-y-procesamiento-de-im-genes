#Integrantes: Luis Carlos Quijada Ceceña
#             Luis Guillermo Flores Ramos
import cv2

def combinar_imagenes(img1_path, img2_path):
    
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        print("Error: No se pudo cargar una o ambas imágenes.")
        return

    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    left_img1 = img1[:, :img1.shape[1]//2]
    right_img1 = img1[:, img1.shape[1]//2:]

    left_img2 = img2_resized[:, :img2_resized.shape[1]//2]
    right_img2 = img2_resized[:, img2_resized.shape[1]//2:]

    combined_img1 = cv2.hconcat([left_img1, right_img2])
    combined_img2 = cv2.hconcat([left_img2, right_img1])

    cv2.imwrite("C:/Users/luisz/Tarea 1 - Sem. Vision P.I/img1-img2.jpg", combined_img1)
    cv2.imwrite("C:/Users/luisz/Tarea 1 - Sem. Vision P.I/img2-img1.jpg", combined_img2)

    print("Imágenes guardadas con éxito.")

if __name__ == "__main__":
    img1_path = "C:/Users/luisz/Downloads/dark_street.jpg"
    img2_path = "C:/Users/luisz/Downloads/checkerboard.png"

    combinar_imagenes(img1_path, img2_path)
