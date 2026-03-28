import cv2
import utils.interpolar
import utils.nitidez


def main():

    # automatizar lectura y escritura de archivos 
    # TODO: IMPLEMENTAR CICLOS PARA PROCESAR TODAS LAS IMAGENES
    DEFAULT_INPUT_DIRECTORY = 'img/'
    DEFAULT_OUTPUT_DIRECTORY = 'results/'

    opcion = int(input("Seleccione una opcion a realizar:\n[1] Interpolar pixeles faltantes\n[2] Mejorar nitidez\n[3] Ambos\n> "))
    filtro = input("Ingrese el filtro aplicado en la imagen del lab 1, es decir imagenX_Y.jpg (el valor X)\n[1] Filtro Negro\n[2] Filtro Blanco\n> ")
    serie = input("Ingrese la serie de la imagen del lab 1, es decir, lab1_imagenX_Y.jpg (el valor Y)\n> ")

    if filtro not in ['1', '2']:
        print("Filtro no valido.")
        return
    
    input_filename = f'lab1_imagen{filtro}_{serie}.png'                 # se obtiene imagen desde la carpeta img/
    interpolar_output_filename = f'lab3_imagen{filtro}_{serie}.png'     # se guarda imagen en la carpeta results/imagenX_Y/ 
    nitidez_output_filename = f'lab3_imagen3_{serie}.png'               # se guarda imagen en la carpeta results/imagen3_X/ 
    

    if opcion == 1:
        # Cargar la imagen con pixeles borrados
        imagen = cv2.imread(f'{DEFAULT_INPUT_DIRECTORY}{input_filename}')       

        # Interpolar los pixeles faltantes
        imagen_interpolada = utils.interpolar.interpolar(imagen)

        # Guardar la imagen interpolada
        cv2.imwrite(f'{DEFAULT_OUTPUT_DIRECTORY}{interpolar_output_filename}', imagen_interpolada)

    elif opcion == 2:

        # Aplicar mejora de nitidez a la imagen interpolada
        c = float(input("Ingrese la constante de amplificación para la mejora de nitidez (ej: 1.5)\n> "))  # Constante de amplificación, se puede ajustar para obtener mejores resultados
        
        imagen_interpolada = cv2.imread(f'{DEFAULT_OUTPUT_DIRECTORY}{interpolar_output_filename}')
        utils.nitidez.nitidez(imagen_interpolada, f'{DEFAULT_OUTPUT_DIRECTORY}{nitidez_output_filename}', c)

    elif opcion == 3:

        # Cargar la imagen con pixeles borrados
        imagen = cv2.imread(f'{DEFAULT_INPUT_DIRECTORY}{input_filename}')       

        # Interpolar los pixeles faltantes
        imagen_interpolada = utils.interpolar.interpolar(imagen)

        # Guardar la imagen interpolada
        cv2.imwrite(f'{DEFAULT_OUTPUT_DIRECTORY}{interpolar_output_filename}', imagen_interpolada)

        # Aplicar mejora de nitidez a la imagen interpolada
        c = float(input("Ingrese la constante de amplificación para la mejora de nitidez (ej: 1.5)\n> "))  # Constante de amplificación, se puede ajustar para obtener mejores resultados
        utils.nitidez.nitidez(imagen_interpolada, f'{DEFAULT_OUTPUT_DIRECTORY}{nitidez_output_filename}', c)

    else:
        print("Opción no válida.")

if __name__ == "__main__":
    main()
