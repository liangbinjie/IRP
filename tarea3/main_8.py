"""
Utilizando un filtro Gausseano con una mascara de 5x5, realize el proceso de suavizado de la imagen5, guarde
la imagen resultante como imagen8.
"""

import cv2

def main():
    # Cargar la imagen de entrada
    img = cv2.imread('output/imagen5.jpg')

    # Aplicar un filtro Gaussiano con una máscara de 5x5
    imagen8 = cv2.GaussianBlur(img, (5, 5), 0)

    # Guardar la imagen resultante
    cv2.imwrite('output/imagen8.jpg', imagen8)
    print("La imagen suavizada se ha guardado en 'output/imagen8.jpg'")
    cv2.imshow('Imagen 8', imagen8)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

