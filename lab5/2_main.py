
"""
Utilizando la biblioteca o herramienta seleccionada, desarrolle una aplicacion que reciba como entradas una
imagen y las constantes complejas a, b, c, d y genere el mapeo directo de la imagen en el Plano w, siempre y
cuando el mapeo exista, guarde la imagen resultante con el nombre imagen2. Use los siguientes valores iniciales
para las constantes complejas: a = 2,1 + j2,1; b = 0; c = 0,003; d = 1 + j.
"""

import utils.mapeo as mapeo
import cv2

def main():
    img = cv2.imread('image.jpg')

    a = complex(2.1, 2.1)
    b = 0
    c = 0.003
    d = complex(1, 1)

    output = mapeo.mapeo_lineal(img, a, b, c, d)

    # cv2.imshow("Original", img)
    # cv2.imshow('output', output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("imagen2.jpg", output)

if __name__ == "__main__":
    main()