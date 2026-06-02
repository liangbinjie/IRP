"""
Este archivo se encarga de implementar el algoritmo KNN (K-Nearest Neighbors)
para clasificar las imágenes de dígitos escritos a mano.

Cada imagen preprocesada (28x28) se representa como un vector de histogramas
de proyeccion horizontal y vertical concatenados (14 caracteristicas con k=4).

El algoritmo KNN funciona de la siguiente manera:
1. Dado un vector de prueba, se compara con los vectores del modelo de entrenamiento.
2. Se calcula la distancia euclidiana entre el vector de prueba y cada vector de entrenamiento.
3. Se seleccionan los K vecinos mas cercanos.
4. Se determina la etiqueta por mayoria de votos entre los K vecinos.
"""

import csv
import math
import os
from collections import Counter

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _cargar_modelo(modelo_h_csv: str, modelo_v_csv: str) -> list[tuple[list[int], int]]:
    """Carga el corpus de entrenamiento concatenando histogramas H+V por fila."""
    corpus = []
    with open(modelo_h_csv, mode='r') as fh, open(modelo_v_csv, mode='r') as fv:
        for row_h, row_v in zip(csv.reader(fh), csv.reader(fv)):
            etiqueta = int(row_h[0])
            vector = list(map(int, row_h[1:])) + list(map(int, row_v[1:]))
            corpus.append((vector, etiqueta))
    return corpus


def compute_confusion_matrix(y_true: list[int], y_pred: list[int], n_classes: int = 10) -> np.ndarray:
    """Construye una matriz de confusion n_classes x n_classes."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        if 0 <= true < n_classes and 0 <= pred < n_classes:
            cm[true, pred] += 1
    return cm


def print_confusion_matrix(cm: np.ndarray) -> None:
    """Imprime la matriz de confusion (filas = real, columnas = prediccion)."""
    print("\nMatriz de confusion (filas = real, columnas = prediccion)\n")
    header = "       " + "  ".join(f"{i:3d}" for i in range(cm.shape[0]))
    print(header)
    print("       " + "-" * (4 * cm.shape[0] + 3))
    for i, row in enumerate(cm):
        print(f"   {i} | " + "  ".join(f"{v:3d}" for v in row))


def save_confusion_matrix_plot(cm: np.ndarray, output_path: str) -> None:
    """Guarda un heatmap de la matriz de confusion como imagen PNG."""
    if not HAS_MATPLOTLIB:
        print("  [AVISO] matplotlib no esta instalado; se omitio el grafico de la matriz.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    n = cm.shape[0]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(range(n), fontsize=11)
    ax.set_yticklabels(range(n), fontsize=11)
    ax.set_xlabel("Prediccion", fontsize=13)
    ax.set_ylabel("Real", fontsize=13)
    ax.set_title("Matriz de confusion — KNN", fontsize=14)

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
    print(f"Matriz de confusion guardada en: {output_path}")


def knn_predecir(vector_prueba: list[int], corpus: list[tuple[list[int], int]], k: int = 3) -> int:
    """
    Dado un vector de prueba y un corpus de entrenamiento, predecir la etiqueta con KNN.
    """
    vecinos = []
    for vector_entrenamiento, etiqueta_entrenamiento in corpus:
        distancia = math.sqrt(sum((p - e) ** 2 for p, e in zip(vector_prueba, vector_entrenamiento)))
        vecinos.append((distancia, etiqueta_entrenamiento))

    vecinos.sort(key=lambda x: x[0])
    k_vecinos = vecinos[:k]
    etiquetas_k_vecinos = [etiqueta for _, etiqueta in k_vecinos]
    return Counter(etiquetas_k_vecinos).most_common(1)[0][0]


def evaluar_modelo(
    test_h_csv: str,
    test_v_csv: str,
    train_h_csv: str,
    train_v_csv: str,
    k: int = 3,
    confusion_matrix_path: str = "output/confusion_matrix.png",
) -> float:
    """
    Evalua la precision del modelo KNN usando histogramas horizontal y vertical concatenados.
    Imprime y guarda la matriz de confusion al finalizar.

    Returns:
        float: Precision en porcentaje, o -1 si hubo error.
    """
    try:
        corpus = _cargar_modelo(train_h_csv, train_v_csv)
    except Exception as e:
        print(f"Error cargando modelo de entrenamiento: {e}.")
        return -1

    y_true: list[int] = []
    y_pred: list[int] = []
    try:
        with open(test_h_csv, mode='r') as fh, open(test_v_csv, mode='r') as fv:
            for row_h, row_v in zip(csv.reader(fh), csv.reader(fv)):
                etiqueta_real = int(row_h[0])
                vector_prueba = list(map(int, row_h[1:])) + list(map(int, row_v[1:]))
                etiqueta_predicha = knn_predecir(vector_prueba, corpus, k)
                y_true.append(etiqueta_real)
                y_pred.append(etiqueta_predicha)
    except Exception as e:
        print(f"Error evaluando el modelo: {e}.")
        return -1

    if not y_true:
        print("No se encontraron datos para evaluar.")
        return -1

    cm = compute_confusion_matrix(y_true, y_pred)
    print_confusion_matrix(cm)
    save_confusion_matrix_plot(cm, confusion_matrix_path)

    correctos = sum(t == p for t, p in zip(y_true, y_pred))
    return (correctos / len(y_true)) * 100
