# Tarea 5 — Reconocimiento de dígitos con Momentos de Hu

Reconocedor estadístico de dígitos (0–9) basado en los 7 invariantes de Hu extraídos de celdas preprocesadas.

## Flujo del proyecto

1. **Preprocesar** (`main.py` opción 1): extrae celdas de las imágenes de entrenamiento en `img/input` y las guarda en `img/output/{0..9}` (sobrescribe la carpeta de salida en cada ejecución).
2. **Entrenar** (opción 2): calcula los Hu Moments de cada celda de entrenamiento, genera CSVs por dígito en `img/hu_output/` y un resumen con media y desviación estándar por dígito en `test/hu_results/hu_moments_summary.csv`.
3. **Generar test** (opción 3): extrae celdas de prueba desde `test/input` hacia `test/output/{0..9}` (también sobrescribe la salida).
4. **Testear** (opción 4):
   - Genera CSVs de Hu Moments de las celdas de prueba en `test/hu_output/`.
   - Clasifica cada celda de `test/output/{0..9}` contra el **modelo de entrenamiento** (`test/hu_results/hu_moments_summary.csv`).
   - Imprime precisión por dígito y guarda la matriz de confusión en `test/hu_results/confusion_matrix.png`.

## Cómo predice un dígito

La lógica de clasificación está en `utils/test.py`, función `predecir_digito`.

### 1. Modelo de referencia

Durante el entrenamiento se construye un perfil por dígito a partir de todas las celdas de `img/output/{dígito}`. Para cada dígito se guardan **7 medias** y **7 desviaciones estándar** (una por cada momento de Hu), en `test/hu_results/hu_moments_summary.csv`.

Cada fila del CSV representa un dígito con esta forma:

```
label, hu1_mean, hu1_std, hu2_mean, hu2_std, ..., hu7_mean, hu7_std
```

### 2. Extracción de Hu Moments de la celda de prueba

Para cada imagen en `test/output/{dígito}`:

1. Se convierte a escala de grises.
2. Se calculan los momentos geométricos con `cv2.moments`.
3. Se obtienen los 7 Momentos de Hu con `cv2.HuMoments`.

El resultado es un vector de 7 valores: `[hu1, hu2, hu3, hu4, hu5, hu6, hu7]`.

### 3. Comparación contra cada dígito del modelo

El vector de la celda se compara **contra las 10 filas** del resumen (dígitos 0–9). Para cada dígito candidato se define una banda de aceptación por momento:

```
banda_i = std_i × PERCENTAGE
rango_i = [mean_i - banda_i, mean_i + banda_i]
```

Donde:

- `mean_i` = media del momento *i* para ese dígito en entrenamiento
- `std_i` = desviación estándar del momento *i* para ese dígito
- `PERCENTAGE` = factor de tolerancia (constante en `utils/test.py`, actualmente `0.4`)

Un momento **coincide** si su valor cae dentro de `rango_i`. Con `PERCENTAGE = 0.4`, las bandas equivalen a **±0.4 desviaciones estándar** alrededor de la media (más estrictas que el rango original ±1σ).

### 4. Regla de decisión

Para cada dígito candidato se calculan dos métricas:

1. **`coincidencias`**: cuántos de los 7 momentos caen dentro de su banda.
2. **`distancia`**: distancia euclidiana normalizada respecto al perfil del dígito:

```
distancia = sqrt( Σ ((hu_i - mean_i) / (std_i + ε))² )
```

(con `ε = 1e-30` para evitar división por cero cuando `std_i ≈ 0`)

**Criterio de selección:**

- **Gana el dígito con más coincidencias.**
- **En empate de coincidencias**, gana el de **menor distancia**.

Ese dígito es la predicción.

### 5. Evaluación

`test_hu_moments` recorre cada carpeta `test/output/{0..9}`, predice el dígito de cada celda y compara con la etiqueta real (el nombre de la carpeta). Al iniciar imprime el rango usado:

```
Rango de clasificación: [avg - std×0.4, avg + std×0.4]
```

Al final reporta:

- Precisión por dígito y global
- Matriz de confusión en consola
- Gráfico `test/hu_results/confusion_matrix.png` (filas = dígito real, columnas = predicción)

## Ejemplo simplificado

Supongamos `PERCENTAGE = 0.4` y una celda con `hu1 = 3.5e-3`.

Para el dígito **8**, si el entrenamiento dio `hu1_mean = 3.55e-3` y `hu1_std = 1.46e-3`:

```
banda = 0.4 × 1.46e-3 = 5.84e-4
rango = [3.55e-3 - 5.84e-4, 3.55e-3 + 5.84e-4]  →  [2.97e-3, 4.13e-3]
```

Como `3.5e-3` está dentro, **hu1 cuenta como coincidencia** para el 8. Se repite para hu2…hu7 y para todos los dígitos; gana quien acumule más coincidencias. Si dos dígitos empatan, gana el de menor distancia normalizada.

## Ajustar la tolerancia

Modifica la constante `PERCENTAGE` en `utils/test.py`:

- **`1.0`**: bandas de ±1σ (equivalente al rango `[mean - std, mean + std]`).
- **`0.4`** (valor actual): bandas más estrictas; menos momentos entran en rango por dígito.
- **Valores más bajos** (ej. `0.3`): aún más estrictas.
- **Valores más altos** (ej. `0.8`): bandas más amplias; más coincidencias, pero mayor riesgo de confundir dígitos parecidos.

La distancia normalizada solo interviene como **desempate**; no sustituye el conteo por bandas.
