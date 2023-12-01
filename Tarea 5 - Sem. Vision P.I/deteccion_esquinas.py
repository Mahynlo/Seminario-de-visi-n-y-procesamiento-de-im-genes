"Integrantes: Luis Guillermo Flores Ramos, Luis Carlos Quijada Ceceña, Malcom Hiram Navarro López, Jose Benjamin Partida Peralta"

import cv2
import numpy as np

def draw_corners(image, corners, color, is_harris=True):
    img = image.copy()
    if is_harris:
        img[corners > 0.01 * corners.max()] = color
    else:
        for i in range(corners.shape[0]):
            cv2.circle(img, (int(corners[i, 0, 0]), int(corners[i, 0, 1])), 3, color, -1)
    return img

grace_hopper_image_path = 'C:/Users/luisz/Downloads/grace-hopper.png'
sudoku_blank_grid_image_path = 'C:/Users/luisz/Downloads/sudoku-blank-grid.png'
grace_hopper_image = cv2.imread(grace_hopper_image_path, cv2.IMREAD_COLOR)
sudoku_blank_grid_image = cv2.imread(sudoku_blank_grid_image_path, cv2.IMREAD_COLOR)

grace_hopper_gray = cv2.cvtColor(grace_hopper_image, cv2.COLOR_BGR2GRAY)
sudoku_blank_grid_gray = cv2.cvtColor(sudoku_blank_grid_image, cv2.COLOR_BGR2GRAY)

grace_hopper_harris = cv2.cornerHarris(grace_hopper_gray, blockSize=2, ksize=3, k=0.04)
sudoku_blank_grid_harris = cv2.cornerHarris(sudoku_blank_grid_gray, blockSize=2, ksize=3, k=0.04)

grace_hopper_shi_tomasi = cv2.goodFeaturesToTrack(grace_hopper_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
sudoku_blank_grid_shi_tomasi = cv2.goodFeaturesToTrack(sudoku_blank_grid_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)

green_color = (0, 255, 0)
grace_hopper_harris_img = draw_corners(grace_hopper_image, grace_hopper_harris, green_color, is_harris=True)
sudoku_blank_grid_harris_img = draw_corners(sudoku_blank_grid_image, sudoku_blank_grid_harris, green_color, is_harris=True)
grace_hopper_shi_tomasi_img = draw_corners(grace_hopper_image, grace_hopper_shi_tomasi, green_color, is_harris=False)
sudoku_blank_grid_shi_tomasi_img = draw_corners(sudoku_blank_grid_image, sudoku_blank_grid_shi_tomasi, green_color, is_harris=False)

cv2.imwrite('grace_hopper_harris.png', grace_hopper_harris_img)
cv2.imwrite('sudoku_blank_grid_harris.png', sudoku_blank_grid_harris_img)
cv2.imwrite('grace_hopper_shi_tomasi.png', grace_hopper_shi_tomasi_img)
cv2.imwrite('sudoku_blank_grid_shi_tomasi.png', sudoku_blank_grid_shi_tomasi_img)


cv2.imshow('grace_hopper_harris.png', grace_hopper_harris_img)
cv2.imshow('sudoku_blank_grid_harris.png', sudoku_blank_grid_harris_img)
cv2.imshow('grace_hopper_shi_tomasi.png', grace_hopper_shi_tomasi_img)
cv2.imshow('sudoku_blank_grid_shi_tomasi.png', sudoku_blank_grid_shi_tomasi_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

