from utils.preprocesar import preprocesar_datos
from utils.entrenar import entrenar_modelo
from utils.knn import evaluar_modelo
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
                print("Generando datos de modelo en 'test/'")
                entrenar_modelo(
                    "test/output",
                    "test/model_data_horizontal.csv",
                    "test/model_data_vertical.csv",
                )
            else:
                print("Generando datos de modelo en 'output/'")
                entrenar_modelo(
                    "img/output",
                    "output/model_data_horizontal.csv",
                    "output/model_data_vertical.csv",
                )
            
        elif opcion == 3:
            k = int(input("Ingrese el valor de k: "))
            print("Predecir digitos...")
            precision = evaluar_modelo(
                "test/model_data_horizontal.csv",
                "test/model_data_vertical.csv",
                "output/model_data_horizontal.csv",
                "output/model_data_vertical.csv",
                k,
            )
            if precision >= 0:
                print(f"Precisión del modelo: {precision:.2f}%")
            
        else:
            print("Opción no válida. Por favor, seleccione una opción del 1 al 4.")
        
        opcion = int(input("[1] Preprocesar datos\n[2] Entrenar modelo\n[3] Evaluar modelo\n[4] Salir\nSeleccione una opción: "))


if __name__ == "__main__":
    main()