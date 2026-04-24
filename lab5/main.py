import cv2
import numpy as np

"""
2. Desarrolle una funcion que reciba como entradas las constantes complejas a, b, c, d y determine si las mismas
genera una funcion de variable compleja cuyo mapeo inverso sí existe.
"""
def tiene_inverso(a, b, c, d):
    # retorna True (verdadero) si es distinto a 0
    return b*c-a*d!=0

"""
3. Desarrolle una funcion que reciba como entradas una imagen y las constantes complejas a, b, c, d y, tomando
la imagen de entrada como el Plano z genere la representacion de dicho plano en el Plano w.
"""
def fn(z, a, b, c, d):
    height, width, _ = z.shape # z es la imagen de entrada, el plano z
    w = np.zeros_like(z) # se hace una copia en negro de z, pues w va a las mismas dimensiones de z
    for y in range(height):
        for x in range(width):
            pixel_value = z[y,x] # opencv accede los datos en forma de imagen[fila, columna]
            z_pixel = complex(x, y) # reprentacion compleja de un pixel en z
            w_pixel = (a*z_pixel + b) / (c*z + d) # aplicamos w = f(z) 
            w_x = int(w_pixel.real) # sacamos el componente real
            w_y = int(w_pixel.imag) # sacamos el componente imaginario
            if 0 <= w_x < width and 0 <= w_y < height: # que no se salga de los bordes
                w[w_y, w_x] = pixel_value
    return w

"""
4. Desarrolle una funcion que reciba como entradas una imagen y las constantes complejas a, b, asuma que
c = 0 ∧ d = 1, genere el mapeo lineal 
"""
def mapeo_lineal(img, a, b, c=0, d=1):
    height, width, _ = img.shape
    output = np.zeros_like(img)             # genera una imagen de salida del mismo tamaño que la original, pero completamente negra
    for y in range(height):                 
        for x in range(width):
            pixel_value = img[y, x]         # obtiene el valor del pixel en la posición (y, x) de la imagen original
            z = complex(x, y)               # convierte las coordenadas del pixel a un número complejo z = x + yi
            w = (a*z + b) / (c*z + d)       # aplica el mapeo lineal para obtener un nuevo número complejo w
            new_x = int(w.real)             # obtiene la parte real de w y la convierte a un entero para usarla como nueva coordenada x
            new_y = int(w.imag)             # obtiene la parte imaginaria de w y la convierte a un entero para usarla como nueva coordenada y
            if 0 <= new_x < width and 0 <= new_y < height:
                output[new_y, new_x] = pixel_value  
    return output

def main():
    img = cv2.imread('image.jpg')

    a = complex(0.8, 0.7)
    b = 100

    # ejecuta pregunta 4
    output = mapeo_lineal(img, a, b)

    cv2.imshow("Original", img)
    cv2.imshow('output', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()