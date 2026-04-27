"""
4. Utilizando el mapeo inverso e interpolacion de pixeles en colindancia N=4, obtenga los valores de los pixeles
faltantes en la imagen generada en el punto 2., guarde la imagen resultante como imagen4.

"""

import cv2
import utils.mapeo as mapeo
import utils.interpolar as interpolar


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
        imagen4 = interpolar.interpolar4(imagen2)

        cv2.imshow("Imagen 4", imagen4)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Guardar la imagen resultante
        cv2.imwrite('output/imagen4.jpg', imagen4)
        print("El mapeo se ha generado y guardado en 'output/imagen4.jpg'")
    else:
        print("El mapeo inverso no existe para las constantes dadas.")

if __name__ == "__main__":
    main()