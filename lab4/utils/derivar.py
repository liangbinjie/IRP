import cv2

def derivarX(image, width, height):
    """
    Deriva la imagen en la direccion X, agarrando el siguiente pixel restando al pixel actual
    """
    image_copy = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convertir a escala de grises para facilitar la derivada

    for y in range(height):
        for x in range(width - 1):
            # Restar el siguiente pixel al pixel actual
            gx = int(gray[y, x+1]) - int(gray[y, x])
            if 255 > gx > 15: # Si el valor de la derivada es mayor a 15 y menor a 255, se considera un borde
                image_copy[y, x] = (0, 0, 0)

    return image_copy

def derivarY(image, width, height):
    """
    Deriva la imagen en la direccion Y, agarrando el siguiente pixel restando al pixel actual
    """
    image_copy = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for y in range(height - 1):
        for x in range(width):
            # Restar el siguiente pixel al pixel actual
            gy = int(gray[y+1, x]) - int(gray[y, x])
            if 255 > gy > 15:
                image_copy[y, x] = (0, 0, 0)

    return image_copy