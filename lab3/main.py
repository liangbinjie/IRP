import cv2
import utils.interpolar


def main():

    input_filename = 'lab1_imagen2_2.jpg'
    output_filename = 'lab3_imagen2_2.jpg'
    
    # Cargar la imagen con pixeles borrados
    imagen = cv2.imread(f'img/{input_filename}')

    # Interpolar los pixeles faltantes
    imagen_interpolada = utils.interpolar.interpolar(imagen)

    # Guardar la imagen interpolada
    cv2.imwrite(f'results/{output_filename}', imagen_interpolada)

if __name__ == "__main__":
    main()
