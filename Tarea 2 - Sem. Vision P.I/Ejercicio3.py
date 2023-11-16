import cv2
import os

def blur_or_sharpen(sigma, weight, image):
    blurred_image = cv2.GaussianBlur(image, (0, 0), sigma)

    sharpened_image = cv2.addWeighted(
        image, 1 + weight, blurred_image, -weight, 0)
    return sharpened_image

image_path = 'C:/Users/luisz/Downloads/Blender_Suzanne1.jpeg'
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Unable to load image at {image_path}")
    exit()

sigma = 10
blur_weight = -1 

blurred_image = blur_or_sharpen(sigma, blur_weight, image)

output_folder = 'C:/Users/luisz/ProcessedImages/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cv2.imwrite(os.path.join(output_folder, 'Blurred_Image.jpeg'), blurred_image)

sharpen_weight = -1  
sharpened_image = blur_or_sharpen(sigma, sharpen_weight, blurred_image)

cv2.imwrite(os.path.join(output_folder, 'Sharpened_Image.jpeg'), sharpened_image)

cv2.imshow('Blurred Image', blurred_image)
cv2.imshow('Sharpened Image', sharpened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



