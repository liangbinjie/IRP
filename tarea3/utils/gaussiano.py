import cv2
import numpy as np

def gaussianFilter5x5(image):
    kernel = np.array([[1, 4, 6, 4, 1],
                       [4,16,24,16, 4],
                       [6,24,36,24, 6],
                       [4,16,24,16, 4],
                       [1, 4, 6, 4, 1]], dtype=np.float32) / 256

    height, width, _ = image.shape
    output = image.copy()

    for y in range(2, height-2):
        for x in range(2, width-2):
            total = np.zeros(3)  # RGB
            
            for ky in range(-2, 3):
                for kx in range(-2, 3):
                    value = image[y + ky][x + kx]
                    weight = kernel[ky + 2][kx + 2]
                    total += value * weight
            
            output[y][x] = total

    return output

image = cv2.imread('../output/imagen3.jpg')
filtered_image = gaussianFilter5x5(image)
cv2.imwrite('../output/gaussian_filtered.jpg', filtered_image)