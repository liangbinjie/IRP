"""
Este programa corresponde a la respuesta 2 del lab05/tarea03

Utilizando la biblioteca o herramienta seleccionada, desarrolle una aplicacion que reciba como entradas una
imagen y las constantes complejas a, b, c, d y genere el mapeo directo de la imagen en el Plano w, siempre y
cuando el mapeo exista, guarde la imagen resultante con el nombre imagen2. Use los siguientes valores iniciales
para las constantes complejas: a = 2,1 + j2,1; b = 0; c = 0,003; d = 1 + j.
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

        # Guardar la imagen resultante
        cv2.imwrite('output/imagen2.jpg', imagen2)
        print("El mapeo se ha generado y guardado en 'output/imagen2.jpg'")
    else:
        print("El mapeo inverso no existe para las constantes dadas.")

if __name__ == "__main__":
    main()
