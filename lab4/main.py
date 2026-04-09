import cv2
import utils.derivar
import utils.superponer

IMAGE_PATH = "img/image.jpg"

def main():
    # Cargar imagen
    image = cv2.imread(IMAGE_PATH)
    width, height = image.shape[1], image.shape[0]

    # Verificar si la imagen se cargó correctamente
    if image is None:
        print("Error: Could not load image.")
        return
    
    image_derivada_x = utils.derivar.derivarX(image, width, height)
    image_derivada_y = utils.derivar.derivarY(image, width, height)

    # Mostrar la imagen original
    cv2.imshow("Original Image", image)
    # Mostrar la imagen derivada en X
    cv2.imshow("Derivada X", image_derivada_x)
    # Mostrar la imagen derivada en Y
    cv2.imshow("Derivada Y", image_derivada_y)

    # Esperar a que se presione una tecla y cerrar las ventanas
    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()