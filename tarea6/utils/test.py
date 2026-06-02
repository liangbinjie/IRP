"""
Objetivo:

Con las celdas extraidas en la carpeta de test/output

Vamos a ir carpeta por carpeta para determinar el porcentaje de reconocimiento correcto de cada numero

Para esto se requiere 
- Generar los Hu Moments de una celda
- Comparar los Hu Moments de cada celda con el resumen (hu_results): por cada fila,
  contar cuántos momentos caen en [mean - std*PERCENTAGE, mean + std*PERCENTAGE] y elegir la fila con más coincidencias
- Determinar si el dígito predicho coincide con el de la carpeta, y llevar un conteo de aciertos y errores
- Al final, imprimir el porcentaje de aciertos para cada carpeta de digito

"""

import os
import csv
import cv2
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

PERCENTAGE = 0.4


def cargar_hu_summary(summary_csv_path: str) -> dict:
    """
    Carga el archivo CSV de resumen de Hu Moments y devuelve un diccionario con
    la estructura: { label: { 'means': [...], 'stds': [...] } }
    """
    hu_summary = {}
    with open(summary_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"]
            means = [float(row[f"hu{i}_mean"]) for i in range(1, 8)]
            stds  = [float(row[f"hu{i}_std"])  for i in range(1, 8)]
            hu_summary[label] = {"means": np.array(means), "stds": np.array(stds)}
    return hu_summary


def extraer_momentos_hu(img) -> np.ndarray:
    """
    Dada una imagen BGR, extrae y devuelve los 7 Momentos de Hu como array numpy.
    """
    if img is None:
        raise ValueError("La imagen proporcionada es None.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments


def predecir_digito(hu_moments: np.ndarray, hu_summary: dict) -> str:
    """
    Dado un vector de 7 Hu Moments y el resumen de medias/desviaciones,
    devuelve la etiqueta (label) cuya fila del resumen coincida con más momentos.

    Para cada fila del resumen, se cuenta cuántos momentos caen dentro del rango
    [mean - std*PERCENTAGE, mean + std*PERCENTAGE]. Gana la fila con más coincidencias; en empate,
    gana la de menor distancia euclidiana normalizada.
    """
    epsilon = 1e-30
    mejor_label = None
    mejor_coincidencias = -1
    mejor_distancia = float("inf")
    margen = PERCENTAGE

    for label, stats in hu_summary.items():
        means = stats["means"]
        stds = stats["stds"]
        banda = stds * margen
        en_rango = (hu_moments >= means - banda) & (hu_moments <= means + banda)
        coincidencias = int(np.sum(en_rango))

        diff = hu_moments - means
        normed = diff / (stds + epsilon)
        distancia = float(np.sqrt(np.sum(normed ** 2)))

        if coincidencias > mejor_coincidencias or (
            coincidencias == mejor_coincidencias and distancia < mejor_distancia
        ):
            mejor_coincidencias = coincidencias
            mejor_distancia = distancia
            mejor_label = label

    return mejor_label


def compute_confusion_matrix(y_true: list[int], y_pred: list[int], n_classes: int = 10) -> np.ndarray:
    """Construye una matriz de confusión n_classes x n_classes."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        if 0 <= true < n_classes and 0 <= pred < n_classes:
            cm[true, pred] += 1
    return cm


def print_confusion_matrix(cm: np.ndarray) -> None:
    """Imprime la matriz de confusión en texto (filas = real, columnas = predicción)."""
    print("\nMatriz de confusión (filas = real, columnas = predicción)\n")
    header = "       " + "  ".join(f"{i:3d}" for i in range(cm.shape[0]))
    print(header)
    print("       " + "-" * (4 * cm.shape[0] + 3))
    for i, row in enumerate(cm):
        print(f"   {i} | " + "  ".join(f"{v:3d}" for v in row))


def save_confusion_matrix_plot(cm: np.ndarray, output_path: str) -> None:
    """Guarda un heatmap de la matriz de confusión como imagen PNG."""
    if not HAS_MATPLOTLIB:
        print("  [AVISO] matplotlib no está instalado; se omitió el gráfico de la matriz.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    n = cm.shape[0]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(range(n), fontsize=11)
    ax.set_yticklabels(range(n), fontsize=11)
    ax.set_xlabel("Predicción", fontsize=13)
    ax.set_ylabel("Real", fontsize=13)
    ax.set_title("Matriz de confusión — Hu Moments", fontsize=14)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=9,
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Matriz de confusión guardada en: {output_path}")


def test_hu_moments(test_dir: str, hu_results_dir: str) -> None:
    """
    Para cada carpeta de dígito (0-9) en *test_dir*, extrae los Hu Moments de
    cada imagen, los compara con el resumen en *hu_results_dir* y reporta el
    porcentaje de aciertos por dígito junto con un resumen global.

    Parámetros
    ----------
    test_dir       : carpeta raíz que contiene subcarpetas 0/, 1/, ..., 9/
    hu_results_dir : carpeta que contiene hu_moments_summary.csv
    """
    summary_csv_path = os.path.join(hu_results_dir, "hu_moments_summary.csv")
    if not os.path.isfile(summary_csv_path):
        print(f"No se encontró el archivo de resumen: {summary_csv_path}")
        return

    hu_summary = cargar_hu_summary(summary_csv_path)

    total_global  = 0
    aciertos_global = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    print(f"\nRango de clasificación: [avg - std×{PERCENTAGE}, avg + std×{PERCENTAGE}]")
    print("\n" + "=" * 50)
    print(f"{'Dígito':<10} {'Aciertos':<12} {'Total':<10} {'Precisión':>10}")
    print("=" * 50)

    for label in range(10):
        str_label = str(label)
        test_label_dir = os.path.join(test_dir, str_label)

        if not os.path.isdir(test_label_dir):
            print(f"[AVISO] No se encontró el directorio: {test_label_dir}")
            continue

        aciertos = 0
        total    = 0

        for cell_img_name in os.listdir(test_label_dir):
            if not cell_img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            cell_img_path = os.path.join(test_label_dir, cell_img_name)
            try:
                img = cv2.imread(cell_img_path)
                if img is None:
                    print(f"  [ERROR] No se pudo leer: {cell_img_path}")
                    continue

                hu_moments  = extraer_momentos_hu(img)
                prediccion  = predecir_digito(hu_moments, hu_summary)

                y_true.append(label)
                y_pred.append(int(prediccion))

                if prediccion == str_label:
                    aciertos += 1
                total += 1

            except Exception as e:
                print(f"  [ERROR] Procesando {cell_img_path}: {e}")

        if total > 0:
            porcentaje = (aciertos / total) * 100
            print(f"{str_label:<10} {aciertos:<12} {total:<10} {porcentaje:>9.1f}%")
        else:
            print(f"{str_label:<10} {'—':<12} {'0':<10} {'N/A':>10}")

        aciertos_global += aciertos
        total_global    += total

    print("=" * 50)
    if total_global > 0:
        precision_global = (aciertos_global / total_global) * 100
        print(f"{'TOTAL':<10} {aciertos_global:<12} {total_global:<10} {precision_global:>9.1f}%")
    else:
        print("No se procesó ninguna imagen.")
    print("=" * 50)

    if y_true:
        cm = compute_confusion_matrix(y_true, y_pred)
        print_confusion_matrix(cm)
        plot_path = os.path.join(hu_results_dir, "confusion_matrix.png")
        save_confusion_matrix_plot(cm, plot_path)

    print()