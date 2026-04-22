import numpy as np
"""
Desarrolle una funcion que reciba como entradas una imagen y las constantes complejas a, b, c, d y, tomando
la imagen de entrada como el Plano z genere la representacion de dicho plano en el Plano w
"""
def mapeo_lineal(img, a, b, c, d):
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

"""
Desarrolle una funcion que reciba como entradas las constantes complejas a, b, c, d y determine si las mismas
genera una funcion de variable compleja cuyo mapeo inverso si existe
"""
def tiene_inverso(a, b, c, d):
    return (b*c - a*d) != 0     # solo sirve si el determinante de la matriz [[a, b], [c, d]] es diferente de cero, lo que garantiza que la función es invertible.


"""
Desarrolle una funcion que reciba como entradas una imagen y las constantes complejas a, b, asuma que
c = 0 ∧ d = 1, genere el mapeo lineal 
"""
def mapeo_lineal2(img, a, b, c=0, d=1):
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