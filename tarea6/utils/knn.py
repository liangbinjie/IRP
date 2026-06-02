"""
Este archivo se encarga de implementar el algoritmo KNN (K-Nearest Neighbors) 
para clasificar las imágenes de dígitos escritos a mano. 

Una vez que se han preprocesado las imágenes y se ha generado el modelo CSV con los datos de cada imagen,
el algoritmo KNN se utilizará para clasificar nuevas imágenes basándose en la similitud de sus características 
con las imágenes del conjunto de entrenamiento.

Sabemos que cada imagen preprocesada es de 28x28 píxeles, lo que significa que cada imagen 
se representa como un vector de 784 características (valores de píxeles).

El algoritmo KNN funciona de la siguiente manera:
1. Dado un directorio de imagenes de prueba, se va folder tras folder, por cada imagen, se vectoriza y se compara con los vectores del modelo CSV.
2. Se calcula la distancia (por ejemplo, la distancia euclidiana) entre el vector de la imagen de prueba y los vectores del modelo CSV.
3. Se seleccionan los K vecinos más cercanos (los K vectores del modelo CSV con la distancia más pequeña).
4. Se determina la etiqueta de la imagen de prueba basándose en la mayoría de las etiquetas de los K vecinos más cercanos.
5. Se devuelve la etiqueta predicha para cada imagen de prueba.
"""

from .entrenar import vectorizar_imagen

def knn_clasificar(imagen_prueba: str, modelo_csv: str, k: int = 3) -> int:
    """
    Clasifica una imagen de prueba utilizando el algoritmo KNN basado en un modelo CSV.
    
    Args:
        imagen_prueba (str): Ruta a la imagen de prueba que se desea clasificar.
        modelo_csv (str): Ruta al archivo CSV que contiene el modelo de entrenamiento.
        k (int): Número de vecinos más cercanos a considerar para la clasificación.
    
    Returns:
        int: La etiqueta predicha para la imagen de prueba (0-9).
    """

    # 1. Vectorizar la imagen de prueba
    vector_prueba = vectorizar_imagen(imagen_prueba)
    if not vector_prueba:
        print(f"No se pudieron obtener los valores de los píxeles para '{imagen_prueba}'.")
        return -1  # Retornar -1 para indicar un error en la vectorización
    
    # 2. Leer el modelo CSV y almacenar los vectores y etiquetas en listas
    import csv
    etiquetas = []
    vectores = []
    try:
        with open(modelo_csv, mode='r') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                etiquetas.append(int(row[0]))  # La primera columna es la etiqueta
                vectores.append(list(map(int, row[1:])))  # Las siguientes columnas son los valores de los píxeles
    except Exception as e:
        print(f"Error leyendo el modelo CSV '{modelo_csv}': {e}.")
        return -1  # Retornar -1 para indicar un error en la lectura del modelo CSV
    
    # 3. Calcular las distancias entre el vector de la imagen de prueba y los vectores del modelo CSV
    import math
    distancias = []
    for i in range(len(vectores)):
        distancia = math.sqrt(sum((vector_prueba[j] - vectores[i][j]) ** 2 for j in range(len(vector_prueba))))
        distancias.append((distancia, etiquetas[i]))
    
    # 4. Seleccionar los K vecinos más cercanos y determinar la etiqueta mayoritaria entre ellos
    distancias.sort(key=lambda x: x[0])  # Ordenar por distancia
    vecinos_cercanos = distancias[:k]  # Obtener los K vecinos más cercanos
    etiquetas_vecinos = [vecino[1] for vecino in vecinos_cercanos]
    etiqueta_predicha = max(set(etiquetas_vecinos), key=etiquetas_vecinos.count)  # Obtener la etiqueta mayoritaria

    # 5. Devolver la etiqueta predicha para la imagen de prueba
    return etiqueta_predicha

