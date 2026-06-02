"""
Funcion para que reciba un directorio de entrada con imagenes preprocesadas de digitos escritos a mano
Dicho directorio debe estar con folders enumerados del 0 al 9, cada folder con imagenes de dicho numero

Objetivo: generar dos modelos CSV (horizontal y vertical) con histogramas de proyeccion
La primera columna es la etiqueta (0-9); las demas cuentan pixeles blancos por franja de k filas/columnas.
"""

import cv2
import numpy as np


def histograma_pixeles(img: np.ndarray, k: int = 4) -> tuple[list[int], list[int]]:
    """
    Cuenta pixeles blancos (255) en franjas horizontales de k filas y franjas verticales de k columnas.

    Returns:
        (h_hist, v_hist): listas de enteros con los conteos por franja.
    """
    white = (img == 255).astype(np.uint8)
    H, W = white.shape
    h_hist = [int(white[i:i + k, :].sum()) for i in range(0, H, k)]
    v_hist = [int(white[:, j:j + k].sum()) for j in range(0, W, k)]
    return h_hist, v_hist


def histograma_desde_ruta(img_path: str, k: int = 4) -> tuple[list[int], list[int]]:
    """
    Lee una imagen en escala de grises y devuelve sus histogramas horizontal y vertical.
    """
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"No se pudo leer la imagen '{img_path}'.")
        return histograma_pixeles(img, k)
    except Exception as e:
        print(f"Error procesando '{img_path}': {e}.")
        return [], []


def entrenar_modelo(
    input_dir: str = "img/output",
    output_horizontal: str = "output/model_data_horizontal.csv",
    output_vertical: str = "output/model_data_vertical.csv",
    k: int = 4,
) -> None:
    """
    Dado un directorio de entrada con imagenes preprocesadas de digitos escritos a mano,
    generar dos CSV con histogramas de proyeccion (horizontal y vertical) por imagen.
    """
    import os
    import csv

    if not os.path.isdir(input_dir):
        print(f"El directorio '{input_dir}' no existe.")
        return

    os.makedirs(os.path.dirname(output_horizontal) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_vertical) or ".", exist_ok=True)

    with open(output_horizontal, mode='w', newline='') as csv_h, \
         open(output_vertical, mode='w', newline='') as csv_v:
        writer_h = csv.writer(csv_h)
        writer_v = csv.writer(csv_v)

        for label in range(10):
            label_dir = os.path.join(input_dir, str(label))
            if not os.path.isdir(label_dir):
                print(f"El directorio '{label_dir}' no existe. Saltando...")
                continue

            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                if not img_path.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"No se pudo leer '{img_path}'. Saltando...")
                        continue

                    h_hist, v_hist = histograma_pixeles(img, k)
                    if not h_hist or not v_hist:
                        print(f"No se pudieron calcular histogramas para '{img_path}'. Saltando...")
                        continue

                    writer_h.writerow([label] + h_hist)
                    writer_v.writerow([label] + v_hist)

                except Exception as e:
                    print(f"Error procesando '{img_path}': {e}. Saltando...")
                    continue

    print(f"Modelo CSV horizontal generado en '{output_horizontal}'.")
    print(f"Modelo CSV vertical generado en '{output_vertical}'.")
