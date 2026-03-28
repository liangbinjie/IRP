import cv2

# # Read image in BGR format (OpenCV default)
image = cv2.imread('src/lab1_imagen2.jpg')


# # Obtener las dimensiones de la imagen
height, width, channels = image.shape

print(f"Image Height: {height} pixels")
print(f"Image Width: {width} pixels")

image_copy = image.copy()

for y in range(height):
    for x in range(width):
        if (x) % 2 == (y) % 2:
            image_copy[y, x] = [0, 0, 0]
        
cv2.imwrite("results/lab1_imagen2_1.png", image_copy)

for y in range(height):
    for x in range(width):
        if (x) % 2 == (y) % 2:
            image_copy[y, x] = [255, 255, 255]  

cv2.imwrite("results/lab1_imagen2_2.png", image_copy)