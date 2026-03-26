import cv2
import utils.interpolar
import utils.nitidez


def main():

    # automatizar lectura y escritura de archivos 
    # TODO: IMPLEMENTAR CICLOS PARA PROCESAR TODAS LAS IMAGENES
    DEFAULT_INPUT_DIRECTORY = 'img/'
    DEFAULT_OUTPUT_DIRECTORY = 'results/'

    filtro = input("Ingrese el filtro aplicado en la imagen del lab 1, es decir imagenX_Y.jpg (el valor X)\n[1] Filtro Negro\n[2] Filtro Blanco\n> ")
    serie = input("Ingrese la serie de la imagen del lab 1, es decir, lab1_imagenX_Y.jpg (el valor Y)\n> ")

    input_filename = f'lab1_imagen{filtro}_{serie}.jpg'                     # se obtiene imagen desde la carpeta img/
    output_filename = f'imagen_{serie}/lab3_imagen{filtro}_{serie}.jpg'     # se guarda imagen en la carpeta results/imagen_X/ con el mismo nombre pero con prefijo lab3_ para indicar que es el resultado del lab 3
    
    # Cargar la imagen con pixeles borrados
    imagen = cv2.imread(f'{DEFAULT_INPUT_DIRECTORY}{input_filename}')       

    # Interpolar los pixeles faltantes
    imagen_interpolada = utils.interpolar.interpolar(imagen)

    # Guardar la imagen interpolada
    cv2.imwrite(f'{DEFAULT_OUTPUT_DIRECTORY}{output_filename}', imagen_interpolada)

    # Aplicar mejora de nitidez a la imagen interpolada
    c = float(input("Ingrese la constante de amplificación para la mejora de nitidez (ej: 1.5)\n> "))  # Constante de amplificación, se puede ajustar para obtener mejores resultados
    nitidez_output_filename = f'lab3_imagen3_{serie}.jpg'
    utils.nitidez.nitidez(imagen_interpolada, f'{DEFAULT_OUTPUT_DIRECTORY}{nitidez_output_filename}', c)

if __name__ == "__main__":
    main()
