from utils.preprocesar import preprocesar_datos
from utils.entrenar import entrenar_modelo
from utils.knn import knn_clasificar
import os

def main():
    opcion = int(input("[1] Preprocesar datos\n[2] Entrenar modelo\n[3] Evaluar modelo\n[4] Salir\nSeleccione una opción: "))
    while opcion != 4:
        if opcion == 1:
            isTest = int(input("¿Desea preprocesar los datos de prueba?\n[0] No, Procesar datos de entrenamiento\n[1] Sí\n> "))
            if isTest == 1:
                preprocesar_datos("test/input", "test/output")
            else:
                preprocesar_datos("img/input", "img/output")
        elif opcion == 2:
            isTest = int(input("¿Desea generar el modelo con los datos de prueba?\n[0] No, generar con datos de entrenamiento\n[1] Sí\n> "))
            if isTest == 1:
                print("Generando datos de modelo en 'test/model_data'")
                entrenar_modelo("test/output", "test/model_data.csv")
            else:
                print("Generando datos de modelo en 'output/model_data'")
                entrenar_modelo("img/output", "output/model_data.csv")
            
        elif opcion == 3:
            print("Predecir digitos...")
            directorio_prueba = "test/output"
            modelo_csv = "output/model_data.csv"
            aciertos = 0
            total = 0

            for folder in sorted(os.listdir(directorio_prueba)):
                folder_path = os.path.join(directorio_prueba, folder)
                if os.path.isdir(folder_path):
                    for img_name in sorted(os.listdir(folder_path)):
                        img_path = os.path.join(folder_path, img_name)
                        if img_path.lower().endswith((".png", ".jpg", ".jpeg")):
                            etiqueta_predicha = knn_clasificar(img_path, modelo_csv, k=3)
                            etiqueta_real = int(folder)
                            if etiqueta_predicha == etiqueta_real:
                                aciertos += 1
                            total += 1
            
            if total > 0:
                precision = (aciertos / total) * 100
                print(f"Precisión del modelo: {precision:.2f}% ({aciertos}/{total} aciertos)")
            else:
                print("No se encontraron imágenes para evaluar.")
            
        else:
            print("Opción no válida. Por favor, seleccione una opción del 1 al 4.")
        
        opcion = int(input("[1] Preprocesar datos\n[2] Entrenar modelo\n[3] Evaluar modelo\n[4] Salir\nSeleccione una opción: "))


if __name__ == "__main__":
    main()