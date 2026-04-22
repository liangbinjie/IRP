""" ===========================================================================

Este programa corresponde a la sección 1.4 del laboratorio 5 de la materia de IRP. 
En esta sección se implementa un mapeo lineal para transformar una imagen utilizando una función de la forma w = (az + b) / (cz + d), donde a, b, c y d son parámetros que controlan la transformación.

Es decir, la imagen de entrada es considerado nuestro plano Z, cada pixel es un número complejo z = x + yi, y la imagen de salida es el plano W, donde cada pixel se obtiene aplicando la función w = (az + b) / (cz + d).

Tests:

1.4.1 - Magnificacion:  a != 0 && b = 0
=> a = 2

1.4.2 - Magnificacion y Rotacion: a = numero complejo && a != 0 && b = 0
=> a = complex(0.8, 0.7)

1.4.3 - Desplazamiento: a = 1 && b != 0 (probar con 100)
=> a = 1, b = 100

1.4.4 - Magnificacion, Rotacion y Desplazamiento: a = numero complejo && a != 0 && b != 0 (probar con 100)
=> a = complex(0.8, 0.7), b = 100

=========================================================================== """

import utils.mapeo as mapeo
import cv2

def main():
    img = cv2.imread('image.jpg')

    a = complex(0.8, 0.7)
    b = 100

    output = mapeo.mapeo_lineal_img(img, a, b)


    cv2.imshow("Original", img)
    cv2.imshow('output', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()