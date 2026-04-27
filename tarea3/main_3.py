"""
Este programa corresponde a la respuesta 3 del lab05/tarea03

Utilizando el mapeo inverso obtenga los valores de los pixeles faltantes en la imagen generada en el punto 2.,
guarde la imagen resultante como imagen3.
"""

import cv2
import utils.mapeo as mapeo

def main():
    # Cargar la imagen de entrada
    img = cv2.imread('image/image.jpg')

    # Definir las constantes complejas
    a = complex(2.1, 2.1)
    b = 0
    c = 0.003
    d = complex(1, 1)

    # Verificar si el mapeo inverso existe
    if mapeo.tiene_inverso(a, b, c, d):
        # Generar el mapeo directo de la imagen en el Plano w
        imagen2 = mapeo.fn(img, a, b, c, d)

        imagen3 = mapeo.mapeo_inverso(imagen2, img, a, b, c, d)

        # Guardar la imagen resultante
        cv2.imwrite('output/imagen3.jpg', imagen3)

        cv2.imshow('Imagen 3', imagen3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("El mapeo se ha generado y guardado en 'output/imagen3.jpg'")
    else:
        print("El mapeo inverso no existe para las constantes dadas.")

if __name__ == "__main__":
    main()