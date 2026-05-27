"""
Funcion para binarizar las celdas de la imagen, con el fin de facilitar su posterior procesamiento.
"""
import cv2

def binarizar(cell):
    """
    Binariza una imagen de celda usando un umbral adaptativo.

    Parámetros
    ----------
    cell : np.ndarray
        Imagen de la celda (en escala de grises).

    Retorna
    -------
    np.ndarray
        Imagen binarizada de la celda.
    """
    # Aplicar un umbral adaptativo para binarizar la imagen
    binary = cv2.adaptiveThreshold(cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return binary