import cv2
import utils.derivar
import utils.superponer

IMAGE_PATH = "img/image3.jpg"

def main():
    # Cargar imagen
    image = cv2.imread(IMAGE_PATH)
    width, height = image.shape[1], image.shape[0]
    print(f"Image dimensions: {width}x{height}")

    if image is None:
        print("Error: Could not load image.")
        return
    
    image_derivada_x = utils.derivar.derivarX(image, width, height)
    image_derivada_y = utils.derivar.derivarY(image, width, height)
    superpuesta = utils.superponer.superponer(image_derivada_x, image_derivada_y, height, width)

    # cv2.imshow("Original Image", image)

    # cv2.imshow("Derivada X", image_derivada_x)

    # cv2.imshow("Derivada Y", image_derivada_y)

    # cv2.imshow("Superpuesta", superpuesta)


    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imwrite("img/derivada_imagen3_x.jpg", image_derivada_x)
    cv2.imwrite("img/derivada_imagen3_y.jpg", image_derivada_y)
    cv2.imwrite("img/superpuesta_imagen3.jpg", superpuesta)


if __name__ == "__main__":
    main()