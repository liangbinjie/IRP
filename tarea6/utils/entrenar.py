"""
Funcion para que reciba un directorio de entrada con imagenes preprocesadas de digitos escritos a mano
Dicho directorio debe estar con folders enumerados del 0 al 9, cada folder con imagenes de dicho numero

Objetivo: generar un modelo CSV con los datos de cada imagen
Dicho modelo CSV va a tener en la primera columna el numero que representa la etiqueta de la imagen (0-9)
Las celdas al ser de 28x28 pixeles, van a tener 784 columnas con los valores de cada pixel (0-255)
"""

def vectorizar_imagen(img_path: str) -> list:
    """
    Dada la ruta de una imagen, leerla en escala de grises, aplanarla a una lista de 784 valores y devolverla.
    """
    import cv2

    try:
        # Leer la imagen en escala de grises
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"No se pudo leer la imagen '{img_path}'.")

        # Aplanar la imagen a una lista de 784 valores
        pixel_values = img.flatten().tolist()
        return pixel_values

    except Exception as e:
        print(f"Error procesando '{img_path}': {e}.")
        return []

def entrenar_modelo(input_dir: str = "img/output", output_file: str = "output/model_data.csv") -> None:
    """
    Dado un directorio de entrada con imagenes preprocesadas de digitos escritos a mano,
    generar un modelo CSV con los datos de cada imagen.
    El modelo CSV tendrá en la primera columna el numero que representa la etiqueta de la imagen (0-9)
    Las celdas al ser de 28x28 pixeles, tendrán 784 columnas con los valores de cada pixel (0-255)
    """
    import os
    import csv

    # Verificar si el directorio de entrada existe
    if not os.path.isdir(input_dir):
        print(f"El directorio '{input_dir}' no existe.")
        return

    # Crear o sobrescribir el archivo CSV de salida
    with open(output_file, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # Recorrer cada carpeta del directorio de entrada (0-9)
        for label in range(10):
            label_dir = os.path.join(input_dir, str(label))
            if not os.path.isdir(label_dir):
                print(f"El directorio '{label_dir}' no existe. Saltando...")
                continue

            # Recorrer cada imagen en la carpeta del label
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                if img_path.lower().endswith((".png", ".jpg", ".jpeg")):
                    try:
                        # Vectorizar la imagen
                        pixel_values = vectorizar_imagen(img_path)
                        if not pixel_values:
                            print(f"No se pudieron obtener los valores de los píxeles para '{img_path}'. Saltando...")
                            continue

                        # Escribir la etiqueta y los valores de los píxeles en el CSV
                        writer.writerow([label] + pixel_values)

                    except Exception as e:
                        print(f"Error procesando '{img_path}': {e}. Saltando...")
                        continue

    print(f"Modelo CSV generado exitosamente en '{output_file}'.")
