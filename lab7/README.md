# Metodologia utilizada

La extracción de celdas se realiza a partir de imágenes que contienen cuadrículas de dígitos manuscritos. El proceso se puede dividir en los siguientes pasos principales:

1.  **Detección de Cajas (Contornos)**:
    *   La imagen de entrada se convierte a escala de grises.
    *   Se aplica el detector de bordes Canny para resaltar los contornos.
    *   Se realiza una dilatación sobre los bordes para conectar componentes cercanos y asegurar que los contornos de las celdas y grupos de celdas estén cerrados.
    *   Se utiliza `cv2.findContours` para encontrar todos los contornos en la imagen.

2.  **Filtrado y Clasificación de Contornos**:
    *   Cada contorno se analiza para determinar si corresponde a una "caja de grupo" (la cuadrícula grande que agrupa 5 dígitos) o a una "caja de celda" (el recuadro de un solo dígito).
    *   La clasificación se basa en el área y la relación de aspecto (ancho/alto) del rectángulo delimitador de cada contorno. Se usan umbrales predefinidos (`GROUP_AREA_MIN`, `GROUP_ASPECT_MIN`, `CELL_AREA_MIN`, etc.) para diferenciar entre grupos, celdas y ruido.

3.  **Deduplicación y Ordenamiento**:
    *   Los contornos detectados a menudo presentan duplicados cercanos. Se aplica un proceso de deduplicación para mantener solo una caja por cada grupo o celda real, eliminando aquellas que están demasiado cerca entre sí (definido por `DEDUP_THRESHOLD`).
    *   Las cajas resultantes (tanto de grupos como de celdas) se ordenan en "orden de lectura": de arriba hacia abajo y de izquierda a derecha. Esto es crucial para asignar las etiquetas correctas a los dígitos.

4.  **Asignación de Etiquetas y Extracción**:
    *   Se determina una "etiqueta base" (0 o 5) a partir del nombre del archivo de imagen. Los archivos con nombres impares comienzan con la etiqueta 0 (para los dígitos 0-4) y los pares con la etiqueta 5 (para los dígitos 5-9).
    *   Se itera sobre cada "caja de celda" detectada. Para cada celda, se identifica a qué "caja de grupo" pertenece basándose en su posición vertical.
    *   La etiqueta final de la celda se calcula sumando el índice del grupo a la etiqueta base. Por ejemplo, si la etiqueta base es 0 y la celda está en el primer grupo (índice 0), su etiqueta será 0. Si está en el segundo (índice 1), será 1, y así sucesivamente.
    *   Se gestiona un índice de archivo para cada etiqueta, de modo que las nuevas celdas se guarden sin sobrescribir las existentes (`cell_NNNN.png`).

5.  **Preprocesamiento y Guardado de Celdas**:
    *   Una vez que una celda es recortada de la imagen original, se somete a una serie de pasos de preprocesamiento para normalizarla:
        1.  Se convierte a escala de grises.
        2.  Se binariza la imagen para obtener un fondo negro y un dígito blanco.
        3.  Se recorta un pequeño margen para eliminar restos de los bordes de la celda.
        4.  El dígito se centra dentro de la imagen resultante.
    *   Finalmente, la imagen de la celda procesada se guarda en la carpeta de salida correspondiente a su etiqueta.

Este proceso se repite para todas las imágenes en el directorio de entrada, extrayendo y clasificando automáticamente todos los dígitos.

## Funciones de Preprocesamiento Detalladas

Las celdas extraídas pasan por dos funciones clave de preprocesamiento antes de ser guardadas: `binarize_cells` y `centrar`.

### `binarize_cells`

Esta función es responsable de convertir la imagen de una celda (en escala de grises) a una imagen binaria (blanco y negro).

*   **Técnica utilizada**: Se emplea un umbral adaptativo gaussiano (`cv2.adaptiveThreshold`). A diferencia de un umbral global simple, esta técnica calcula el umbral para cada píxel basándose en los valores de los píxeles vecinos.
*   **Parámetros**:
    *   `cv2.ADAPTIVE_THRESH_GAUSSIAN_C`: El valor del umbral es una suma ponderada de los valores de la vecindad (gaussiana).
    *   `cv2.THRESH_BINARY_INV`: Se invierte el resultado. Los píxeles por encima del umbral se establecen en 0 (negro) y los que están por debajo en 255 (blanco). Esto asegura que el dígito sea blanco y el fondo negro.
*   **Propósito**: Este método es robusto a diferentes condiciones de iluminación en la imagen, asegurando una binarización consistente y limpia de cada dígito.

### `centrar`

El propósito de esta función es normalizar cada dígito, centrándolo y escalándolo en un lienzo de tamaño fijo (por defecto, 28x28 píxeles), similar al formato del conjunto de datos MNIST.

1.  **Asegurar Binarización**: Primero, se asegura de que la imagen de entrada sea puramente binaria.
2.  **Encontrar Contenido**: Se localizan todos los píxeles blancos (el dígito) y se calcula su rectángulo delimitador (`boundingRect`) para recortar el dígito sin márgenes innecesarios.
3.  **Calcular Escala**: Se determina el factor de escala para que la dimensión más grande del dígito (ancho o alto) se ajuste a un tamaño objetivo (`size - 2 * padding`), preservando la relación de aspecto original.
4.  **Redimensionar**: El dígito recortado se redimensiona a las nuevas dimensiones calculadas utilizando la interpolación `cv2.INTER_NEAREST`, que es adecuada para imágenes binarias sin generar tonos grises.
5.  **Crear Lienzo**: Se crea un lienzo cuadrado, completamente negro, del tamaño final deseado (ej. 28x28).
6.  **Pegar y Centrar**: Se calculan las coordenadas para pegar el dígito redimensionado en el centro del lienzo. Finalmente, el dígito se "pega" en su posición centrada.

*   **Propósito**: Este proceso garantiza que todas las imágenes de los dígitos tengan un formato uniforme, con el dígito centrado y de un tamaño consistente, lo cual es fundamental para el entrenamiento de modelos de reconocimiento de patrones.
