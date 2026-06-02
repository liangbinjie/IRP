# Tarea 6 — Reconocimiento de dígitos con histogramas y KNN

Sistema para reconocer dígitos escritos a mano a partir de fotografías de hojas con celdas. 

Flujo:

1. **Preprocesar** las imágenes (detectar celdas, binarizar, centrar). Preprocesar las imagenes de modelo y de prueba.
2. **Entrenar** un modelo extrayendo histogramas de proyección horizontal y vertical. Hacerlo con los del modelo y de prueba.
3. **Evaluar** con KNN y generar una matriz de confusión.

---

## Requisitos

- Python 3.10 o superior (recomendado)
- pip

### Dependencias de Python

| Paquete | Uso |
|---------|-----|
| `opencv-python` | Lectura de imágenes, detección de celdas, binarización |
| `numpy` | Operaciones con matrices e histogramas |
| `matplotlib` | Gráfico PNG de la matriz de confusión (opcional pero recomendado) |

Instalación:

```bash
cd tarea6
pip install -r requierements.txt
```

Si `matplotlib` no está instalado, la evaluación sigue funcionando: se imprime la matriz de confusión en consola, pero no se guarda el PNG.

---

## Uso

Desde la carpeta `tarea6`, ejecuta:

```bash
python main.py
```

Aparecerá un menú interactivo:

```
[1] Preprocesar datos
[2] Entrenar modelo
[3] Evaluar modelo
[4] Salir
```

### Opción 1 — Preprocesar datos

Extrae y normaliza cada celda de las hojas fotografiadas.

| Respuesta | Acción |
|-----------|--------|
| `0` | Procesa `img/input/` → `img/output/` |
| `1` | Procesa `test/input/` → `test/output/` |

> **Nota:** el directorio de salida se borra y recrea en cada ejecución.

### Opción 2 — Entrenar modelo

Genera los archivos CSV con las características de cada celda.

| Respuesta | Salida |
|-----------|--------|
| `0` | `output/model_data_horizontal.csv` y `output/model_data_vertical.csv` |
| `1` | `test/model_data_horizontal.csv` y `test/model_data_vertical.csv` |

Cada fila del CSV tiene la forma `[etiqueta, bin0, bin1, …]`. Con imágenes de 28×28 y `k=4` (tamaño de franja por defecto), hay **7 bins horizontales** y **7 bins verticales** por imagen.

### Opción 3 — Evaluar modelo

Clasifica las celdas de prueba con **KNN** comparándolas contra el modelo de entrenamiento.

1. Ingresa el valor de **k** (número de vecinos más cercanos, por ejemplo `3` o `5`).
2. El programa muestra la **precisión** en porcentaje.
3. Imprime la **matriz de confusión** en consola (filas = etiqueta real, columnas = predicción).
4. Guarda el gráfico en `output/confusion_matrix.png`.

Usa los CSV de entrenamiento en `output/` y los de prueba en `test/`.

### Flujo recomendado

```
1. Colocar hojas de entrenamiento en img/input/
2. Menú → [1] → 0  (preprocesar entrenamiento)
3. Menú → [2] → 0  (generar modelo)
4. Colocar hojas de prueba en test/input/
5. Menú → [1] → 1  (preprocesar prueba)
6. Menú → [2] → 1  (generar CSV de prueba)
7. Menú → [3]      (evaluar con el k deseado)
```

---

## Cómo funciona el modelo

Cada celda de 28×28 se convierte en un vector de **14 características**:

- **7 histogramas horizontales:** cuenta de píxeles blancos en franjas de 4 filas.
- **7 histogramas verticales:** cuenta de píxeles blancos en franjas de 4 columnas.

KNN calcula la distancia euclidiana entre el vector de prueba y todos los vectores de entrenamiento, selecciona los **k** vecinos más cercanos y predice la etiqueta por mayoría de votos.

---

## Consejos para mejorar la precisión

- Usar **más imágenes de entrenamiento**; con pocos datos la precisión es inestable.
- Escribe los dígitos con trazo **claro y completo**; trazos débiles pueden perderse en el preprocesamiento.
- Aumenta el **DPI** de la fotografía si los trazos se ven borrosos.
- Prueba distintos valores de **k** en la evaluación; valores mayores pueden ayudar, pero si el conjunto de entrenamiento es pequeño o ruidoso, la precisión puede estancarse o empeorar.
- Algunos dígitos se confunden con frecuencia (por ejemplo, 3 con 7, 6 con 1, 8 con otros). Revisa la matriz de confusión para identificar estos casos.

---
