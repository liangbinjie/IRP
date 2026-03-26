import cv2

image = cv2.imread('src/lab1_imagen2.jpg')

# Obtener las dimensiones de la imagen
height, width, channels = image.shape

print(f"Image Height: {height} pixels")
print(f"Image Width: {width} pixels")

for y in range(height):
    for x in range(width):
        if (x) % 2 == (y) % 2:
            image[y, x] = [0, 0, 0] 

cv2.imwrite("results/lab1_imagen1_2.jpg", image)

for y in range(height):
    for x in range(width):
        if (x) % 2 == (y) % 2:
            image[y, x] = [255, 255, 255]  

cv2.imwrite("results/lab1_imagen2_2.jpg", image)