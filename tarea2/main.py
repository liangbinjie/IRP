import cv2
import utils.derivar
import utils.superponer
import time


IMAGE_PATH = "img/image2.jpg"

def main():
    # Cargar imagen
    image = cv2.imread(IMAGE_PATH)
    width, height = image.shape[1], image.shape[0]
    print(f"Image dimensions: {width}x{height}")

    if image is None:
        print("Error: Could not load image.")
        return
    
    """
    Sobel Function: funciona basicamente como las funciones de derivadas que realizamos, donde se puede definir la direccion de la derivada, el tamaño del kernel, etc.
        - image: Primer parametro es la imagen de entrada
        - ddepth: Profundidad de la imagen de salida. Si se establece en cv2.CV_64F, la imagen de salida tendrá una profundidad de 64 bits en coma
        - dx: Orden de la derivada en la dirección x (0, 1, 2, ...)
        - dy: Orden de la derivada en la dirección y (0, 1, 2, ...)
        - ksize: Tamaño del kernel utilizado para la derivación (opcional, por defecto es 3)
    """
    # sobel_derivada_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # X direction
    # sobel_derivada_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)  # Y direction

    
    """Seccion de derivada y superposicion manual"""
    f1_start_time = time.time() # iniciar tiempo de ejecucion de las funciones de derivada y superposicion
    image_derivada_x = utils.derivar.derivarX(image, width, height)
    image_derivada_y = utils.derivar.derivarY(image, width, height)
    superpuesta = utils.superponer.superponer(image_derivada_x, image_derivada_y, height, width)
    f1_end_time = time.time() # finalizar tiempo de ejecucion de las funciones de derivada y superposicion
    print("--- Manual: %s seconds ---" % (f1_end_time - f1_start_time))
    
    """
    Canny Function
        - image: Primer parametro es la imagen de entrada
        - threshold1: Define el umbral inferior para el proceso de histéresis (es decir si la gradiente es menor que este valor, se descarta como borde)
        - threshold2: Define el umbral superior para el proceso de histéresis (es decir si la gradiente es mayor que este valor, se considera un borde fuerte)
        - edges: Es la imagen de salida que contiene los bordes detectados. Es una imagen binaria donde los píxeles de borde se establecen en 255 (blanco) y los demás píxeles se establecen en 0 (negro)
        - appertureSize: Tamaño del kernel utilizado para la detección de bordes (opcional, por defecto es 3)
        - L2gradient: Si se establece en True, se utiliza la norma L2 para calcular la magnitud del gradiente. Si es False, se utiliza la norma L1 (opcional, por defecto es False)
    """    

    f2_start_time = time.time() # iniciar tiempo de ejecucion de la funcion Canny
    canny = cv2.Canny(image, 100, 200)
    f2_end_time = time.time() # finalizar tiempo de ejecucion de la funcion Canny
    print("--- Canny: %s seconds ---" % (f2_end_time - f2_start_time))

    # cv2.imshow("Original Image", image)

    # cv2.imshow("Sobel X", sobel_derivada_x)
    # cv2.imshow("Sobel Y", sobel_derivada_y)

    # cv2.imshow("Derivada X", image_derivada_x)
    # cv2.imshow("Derivada Y", image_derivada_y)

    cv2.imshow("Canny", canny)

    cv2.imshow("Superpuesta", superpuesta)


    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()