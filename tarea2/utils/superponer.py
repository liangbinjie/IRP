
def superponer(imagenX, imagenY, height, width):
    """
    Superpone las dos imagenes derivadas
    """
    superpuesta = imagenX.copy()
    for y in range(height):
        for x in range(width):
            superpuesta[y, x] = (imagenX[y, x] + imagenY[y, x])

    return superpuesta


def superponerOriginal(imagenOriginal, superpuesta, height, width):
    """
    Superpone la imagen original con la imagen superpuesta de las derivadas
    """
    resultado = imagenOriginal.copy()
    for y in range(height):
        for x in range(width):
            resultado[y, x] = (imagenOriginal[y, x] + superpuesta[y, x])
    
    return resultado