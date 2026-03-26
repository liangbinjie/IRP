'''
un programa que reciba como entrada una imagen con pixeles borrados del Lab1 y
utilizando interpolacion con colindancia 4 recupera la informacion de los pixeles faltantes.
'''

'''
Overflow de uint8: Al sumar valores de pixeles (que son uint8) para calcular el promedio, 
es posible que la suma exceda el rango de 0-255, lo que puede causar un overflow. 
Para evitar esto, se convierte cada valor de pixel a int antes de acumular la suma, 
y luego se asegura de que el resultado final esté dentro del rango válido al convertirlo nuevamente a uint8.

Ejemplo de overflow:
suma_r = 0
for v in vecinos:
    suma_r += v[0]  # Si v[0] es 255 y hay suficientes vecinos, suma_r puede exceder 255, causando un overflow.
'''

def promedio(vecinos):
    # Promedia una lista de pixeles RGB vecinos (colindancia 4).
    n = len(vecinos)

    if n == 0:
        return [0, 0, 0]

    suma_r = 0
    suma_g = 0
    suma_b = 0
    
    for v in vecinos:
        # Se convierte a int para evitar overflow con uint8 al acumular. 
        suma_r += int(v[0])
        suma_g += int(v[1])
        suma_b += int(v[2])

    r = max(0, min(255, int(round(suma_r / n))))
    g = max(0, min(255, int(round(suma_g / n))))
    b = max(0, min(255, int(round(suma_b / n))))
    
    return [r, g, b]

def interpolar(imagen):
    # obtener las dimensiones de la imagen
    filas, columnas = imagen.shape[:2]

    # Trabaja sobre una copia para no modificar la imagen de entrada.
    imagen_recuperada = imagen.copy()

    for y in range(filas):
        for x in range(columnas):
            if (x) % 2 == (y) % 2:                  # pixel faltante
                # Recolecta vecinos validos (arriba, abajo, izquierda, derecha).
                vecinos = []
                if y > 0:
                    vecinos.append(imagen[y-1, x])  # vecino superior
                if y < filas - 1:
                    vecinos.append(imagen[y+1, x])  # vecino inferior
                if x > 0:
                    vecinos.append(imagen[y, x-1])  # vecino izquierdo
                if x < columnas - 1:
                    vecinos.append(imagen[y, x+1])  # vecino derecho
                
                pixel = promedio(vecinos)  # promedio de los vecinos
                imagen_recuperada[y, x] = pixel
                
    return imagen_recuperada
