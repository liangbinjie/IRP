"""
Clasificador de imagenes de perros y gatos
=====================================================
Estructura de los folders:
    dataset/
        dogs/   <- imagenes de perros de entrenamiento
        cats/   <- imagenes de gatos de entrenamiento
    test/
        dogs/   <- imagenes de perros de prueba
        cats/   <- imagenes de gatos de prueba

Uso:
    python test.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ── Configuracion ──────────────────────────────────────────────────────────────────
DATASET_DIR = "dataset"
TEST_DIR    = "test"
IMG_SIZE    = (128, 128)
# ──────────────────────────────────────────────────────────────────────────────


# ── Cargar imagenes ─────────────────────────────
def load_images(dataset_dir):
    X, y = [], []
    classes = {"cats": 0, "dogs": 1}

    for class_name, label in classes.items():
        folder = os.path.join(dataset_dir, class_name)
        print(f"Loading '{class_name}' from {folder} ...")

        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            try:
                img      = imread(filepath)
                img      = resize(img, IMG_SIZE)
                img_gray = rgb2gray(img)
                features = img_gray.flatten()
                X.append(features)
                y.append(label)
            except Exception as e:
                print(f"  Saltando {filename}: {e}")

    return np.array(X), np.array(y)


X, y = load_images(DATASET_DIR)
print(f"\nImagenes cargadas : {len(X)}")
print(f"Cats : {(y == 0).sum()}  |  Dogs : {(y == 1).sum()}\n")


# ── Escalar las imagenes ──────────────────────────────────────────────────────
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ── Entrenar el SVM en los datos completos ────────────────────────────────────
print("Entrenando SVM ...")
svm = SVC(kernel="rbf", C=1.0, gamma="scale")
svm.fit(X_scaled, y)
print("Entrenamiento completo!\n")

# Evaluación rápida en un split hold-out
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
svm_eval = SVC(kernel="rbf", C=1.0, gamma="scale")
svm_eval.fit(X_train, y_train)
y_pred = svm_eval.predict(X_test)
print("── Evaluación (20% hold-out) ──")
print(classification_report(y_test, y_pred, target_names=["cat", "dog"]))

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["cat", "dog"],
    cmap="Blues", ax=ax
)
ax.set_title("Matriz de confusión")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)
plt.show()


# ── Visualizar el hiperplano via PCA (proyeccion 2D) ─────────────────────────
print("Proyectando a 2D con PCA para visualizar el hiperplano ...")

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)

# Entrenar un nuevo SVM en la proyeccion 2D para que el hiperplano viva en 2D
svm_2d = SVC(kernel="rbf", C=1.0, gamma="scale")
svm_2d.fit(X_2d, y)

# Construir una malla de puntos en el espacio 2D
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400)
)
Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Graficar
colors = {0: "#4A90D9", 1: "#E8734A"}   # blue = cat, orange = dog
fig, ax = plt.subplots(figsize=(8, 6))

# Regiones de decision
ax.contourf(xx, yy, Z, alpha=0.25, levels=[-0.5, 0.5, 1.5],
            colors=["#4A90D9", "#E8734A"])
# Frontera de decision
ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=1.5,
           linestyles="--")

# Graficar puntos
for label, name, color in [(0, "cat", "#1a5fa8"), (1, "dog", "#b84a1e")]:
    mask = y == label
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=color, label=name, alpha=0.7, edgecolors="white",
               linewidths=0.4, s=40)

# Vectores de soporte (proyectados)
sv_2d = X_2d[svm_2d.support_]
ax.scatter(sv_2d[:, 0], sv_2d[:, 1],
           facecolors="none", edgecolors="black", linewidths=1.2,
           s=80, label="Vectores de soporte")

ax.set_title("Frontera de decisión del SVM (proyección 2D PCA)", fontsize=13)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)")
ax.legend()
plt.tight_layout()
plt.savefig("hyperplane.png", dpi=120)
print("Gráfico del hiperplano guardado → hyperplane.png\n")
plt.show()


# ── Predecir todas las imagenes en test/ ──────────────────────────────────────
def preprocess(image_path):
    img      = imread(image_path)
    img      = resize(img, IMG_SIZE)
    img_gray = rgb2gray(img)
    return img_gray.flatten()

def predict_folder(test_dir):
    classes   = {"cats": 0, "dogs": 1}
    all_imgs  = []   # (ruta, etiqueta real)

    for class_name, label in classes.items():
        folder = os.path.join(test_dir, class_name)
        if not os.path.isdir(folder):
            print(f"  Folder not found: {folder}")
            continue
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            all_imgs.append((filepath, label))

    if not all_imgs:
        print("No test images found.")
        return

    print(f"Prediciendo {len(all_imgs)} imagen(es) de prueba ...\n")

    # Disposicion en grid
    n      = len(all_imgs)
    ncols  = min(n, 4)
    nrows  = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3, nrows * 3 + 0.5))
    axes = np.array(axes).flatten()

    correct = 0   # correctas
    for i, (filepath, true_label) in enumerate(all_imgs):
        try:
            feat  = preprocess(filepath)
            feat  = scaler.transform(feat.reshape(1, -1))
            pred  = svm.predict(feat)[0]
            plabel = "dog" if pred == 1 else "cat"
            tlabel = "dog" if true_label == 1 else "cat"
            ok     = pred == true_label
            if ok:
                correct += 1

            img = imread(filepath)
            axes[i].imshow(img)
            axes[i].axis("off")
            color = "green" if ok else "red"
            axes[i].set_title(
                f"Pred: {plabel}\nTrue: {tlabel}",
                fontsize=9, color=color
            )
            print(f"  {os.path.basename(filepath):30s}  pred={plabel:4s}  true={tlabel:4s}  {'✓' if ok else '✗'}")
        except Exception as e:
            axes[i].axis("off")
            print(f"  Error en {filepath}: {e}")

    # Ocultar ejes no utilizados
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    accuracy = correct / len(all_imgs) * 100
    fig.suptitle(f"Predicciones en prueba — precisión: {accuracy:.1f}%", fontsize=12)
    plt.tight_layout()
    plt.savefig("test_predictions.png", dpi=120)
    print(f"\nPrecisión en conjunto de prueba: {correct}/{len(all_imgs)} = {accuracy:.1f}%")
    print("Malla guardada → test_predictions.png")
    plt.show()  


predict_folder(TEST_DIR)