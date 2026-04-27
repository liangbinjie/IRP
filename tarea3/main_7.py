"""
Utilizando un filtro Gausseano con una mascara de 5x5, realize el proceso de suavizado de la imagen4, guarde
la imagen resultante como imagen7.
"""

import cv2

def main():
    image = cv2.imread('output/imagen4.jpg')
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.imshow('output/imagen7.jpg', blurred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('output/imagen7.jpg', blurred)
if __name__ == "__main__":
    main()