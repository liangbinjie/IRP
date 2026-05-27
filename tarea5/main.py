import utils.preprocesar as preprocesar
from utils.test import test_hu_moments
import utils.hu as hu

HU_INPUT_DIR = "img/output"
HU_OUTPUT_DIR = "img/hu_output"
HU_RESULTS = "test/hu_results"
TEST_OUTPUT_DIR = "test/output"
TEST_INPUT_DIR = "test/input"

def main():
    opcion = int(input("Seleccione una opción:\n1. Preprocesar\n2. Probar\n3. Generar Archivos Test\n4. Testear\n\n"))
    if opcion == 1:
        print("Preprocesar el modelo...")
        preprocesar.preprocesar_imagen()
    elif opcion == 2:
        print("Entrenar el modelo...")
        for i in range(10):
            label = str(i)
            input_dir = f"{HU_INPUT_DIR}/{label}"
            hu.generar_csv_hu(input_dir, HU_OUTPUT_DIR, label)
            hu.generar_promedio_hu(HU_OUTPUT_DIR, HU_RESULTS)

    elif opcion == 3:
        print("Generando test...")
        preprocesar.preprocesar_imagen(TEST_INPUT_DIR, TEST_OUTPUT_DIR)

    elif opcion == 4:
        print("Testeando...")
        for i in range(10):
            label = str(i)
            input_dir = f"test/output/{label}"
            hu.generar_csv_hu(input_dir, "test/hu_output", label)
            hu.generar_promedio_hu("test/hu_output", "test/hu_output")
        # test_hu_moments(TEST_OUTPUT_DIR, HU_RESULTS)
        test_hu_moments("test/output", "test/hu_results")

    else:
        print("Opción no válida. Por favor, seleccione 1, 2 o 3.")
    
if __name__ == "__main__":
    main()