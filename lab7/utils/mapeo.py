import numpy as np

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