import cv2
import os
import numpy as np


"""
Se va a procesar todas las celdas que tiene cada carpeta de digito
Para cada imagen se va a extraer los momentos de Hu
Se va a guardar un archivo csv con el nombre de la imagen y los momentos de Hu
Formato: nombre_imagen, hu1, hu2, hu3, hu4, hu5, hu6, hu7
"""

def extraer_momentos_hu(img) -> list[float]:
    """
    Dada una imagen de una celda, extraer los momentos de Hu y devolverlos como una lista de 7 valores.
    """

    if img is None:
        raise ValueError("La imagen proporcionada es None. Asegúrese de que la ruta sea correcta y el archivo exista.")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments.tolist()

def generar_csv_hu(input_dir: str, output_dir: str, label: str) -> None:
    """
    Dado un directorio de entrada con imágenes de celdas, extraer los momentos de Hu
    y guardarlos en un archivo CSV en el directorio de salida.
    """

    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, f"hu_moments_{label}.csv")

    with open(output_csv_path, "w") as f:
        for img in os.listdir(input_dir):
            if img.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(input_dir, img)
                try:
                    image = cv2.imread(img_path)
                    hu_values = extraer_momentos_hu(image)
                    csv_line = f"{img}," + ",".join(f"{v:.6e}" for v in hu_values) + "\n"
                    f.write(csv_line)
                except Exception as e:
                    print(f"Error procesando {img_path}: {e}")

"""
Generar un promedio con desviacion de los momentos de Hu de cada carpeta de digito
Guardarlo en HU_RESULTS
"""
def generar_promedio_hu(input_dir: str, output_dir: str) -> None:
    """
    Dado un directorio de entrada con archivos CSV de momentos de Hu, calcular el promedio y desviación
    de cada momento para cada etiqueta y guardarlo en un archivo CSV en el directorio de salida.
    """

    hu_data = {}
    for csv_file in os.listdir(input_dir):
        if csv_file.startswith("hu_moments_") and csv_file.endswith(".csv"):
            label = csv_file[len("hu_moments_"):-len(".csv")]
            with open(os.path.join(input_dir, csv_file), "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) == 8:  # nombre_imagen + 7 momentos
                        hu_values = list(map(float, parts[1:]))
                        if label not in hu_data:
                            hu_data[label] = []
                        hu_data[label].append(hu_values)

    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, "hu_moments_summary.csv")
    with open(output_csv_path, "w") as f:
        f.write("label,hu1_mean,hu1_std,hu2_mean,hu2_std,hu3_mean,hu3_std,"
                "hu4_mean,hu4_std,hu5_mean,hu5_std,hu6_mean,hu6_std,"
                "hu7_mean,hu7_std\n")
        for label, values in hu_data.items():
            values_array = np.array(values)
            means = np.mean(values_array, axis=0)
            stds = np.std(values_array, axis=0)
            line = f"{label}," + ",".join(f"{m:.6e},{s:.6e}" for m, s in zip(means, stds)) + "\n"
            f.write(line)