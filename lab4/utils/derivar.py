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
            gx = abs(int(gray[y, x+1]) - int(gray[y, x]))
            image_copy[y, x] = gx

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
            gy = abs(int(gray[y+1, x]) - int(gray[y, x]))
            image_copy[y, x] = gy

    return image_copy

def derivarXBGR(imagen, width, height):
    """
    Deriva la imagen en la direccion X, agarrando el siguiente pixel restando al pixel actual
    """
    image_copy = imagen.copy()

    for y in range(height):
        for x in range(width - 1):
            # Restar el siguiente pixel al pixel actual
            for c in range(3):  # Para cada canal de color (B, G, R)
                gx = abs(int(imagen[y, x+1, c]) - int(imagen[y, x, c]))
                image_copy[y, x, c] = gx

    return image_copy

def derivarYBGR(imagen, width, height):
    """
    Deriva la imagen en la direccion Y, agarrando el siguiente pixel restando al pixel actual
    """
    image_copy = imagen.copy()
    for y in range(height - 1):
        for x in range(width):
            # Restar el siguiente pixel al pixel actual
            for c in range(3):  # Para cada canal de color (B, G, R)
                gy = abs(int(imagen[y+1, x, c]) - int(imagen[y, x, c]))
                image_copy[y, x, c] = gy

    return image_copy