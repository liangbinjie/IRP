import cv2
import os
import shutil

from . import extraer

def obtener_etiqueta(filename):
    """Si el nombre del archivo termina en un dígito impar (1, 3, 5, 7, 9), devuelve True"""
    stem   = os.path.splitext(os.path.basename(filename))[0]
    digits = [c for c in stem if c.isdigit()]
    if not digits:
        raise ValueError(f"No digit found in filename: {filename}")
    return int(digits[-1]) % 2 == 1

def preprocesar_imagen(input_dir: str = "img/input", 
                       output_dir: str = "img/output") -> None:
    """
    Dada un directorio de entrada con imagenes, aplicar un preprocesamiento
    Guardarlas en un directorio de salida.
    """

    # establecemos una lista de las rutas de archivos del directorio de entrada
    image_paths = sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    if not image_paths:
        print(f"No hay imagenes en '{input_dir}'")
        return

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    total_celdas_guardadas = 0
    for img_path in image_paths:
        try:
            img = cv2.imread(img_path)
            isOdd = obtener_etiqueta(img_path)
            extracted_cells = extraer.extraer_celdas(img, output_dir, isOdd)
            print(f"Procesada {img_path} → {extracted_cells} celdas guardadas")

            total_celdas_guardadas += extracted_cells

        except Exception as e:
            print(f"Error leyendo {img_path}: {e}")
            continue

    print(f"Total de celdas guardadas: {total_celdas_guardadas}")
