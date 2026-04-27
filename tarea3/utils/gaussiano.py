import cv2
import numpy as np

def gaussianFilter5x5(image):
    kernel = np.array([[1, 4, 6, 4, 1],
                       [4,16,24,16, 4],
                       [6,24,36,24, 6],
                       [4,16,24,16, 4],
                       [1, 4, 6, 4, 1]], dtype=np.float32) / 256

    height, width, channels = image.shape
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

def gaussianFilter5x5_fast(image):
    kernel_1d = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16

    height, width, channels = image.shape
    
    temp = np.zeros_like(image, dtype=np.float32)
    output = np.zeros_like(image, dtype=np.float32)

    # Horizontal pass
    for y in range(height):
        for x in range(2, width-2):
            total = np.zeros(3)
            for k in range(-2, 3):
                total += image[y][x + k] * kernel_1d[k + 2]
            temp[y][x] = total

    # Vertical pass
    for y in range(2, height-2):
        for x in range(width):
            total = np.zeros(3)
            for k in range(-2, 3):
                total += temp[y + k][x] * kernel_1d[k + 2]
            output[y][x] = total

    return output.astype(np.uint8)

image = cv2.imread('../output/imagen3.jpg')
filtered_image = gaussianFilter5x5_fast(image)
cv2.imwrite('../output/gaussian_filtered_fast.jpg', filtered_image)