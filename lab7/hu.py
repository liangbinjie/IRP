import cv2
import utils.mapeo as mapeo

def hu_moments(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(img)
    hu = cv2.HuMoments(moments)
    return hu

# image = cv2.imread("img/hu_input/cell_0016.png");
# image2 = cv2.imread("img/hu_input/cell_0017.png");

# # imagen mas pequeña
# a = 0.5
# b = 0

# image1 = mapeo.mapeo_lineal(image, a, b)

# # imagen con desplazamiento
# a = 1
# b = 6

# image2 = mapeo.mapeo_lineal(image, a, b)

# image3 = mapeo.mapeo_lineal(image, a, b)

# cv2.imshow("Original", image)
# cv2.imshow("Imagen 50%", image1)
# cv2.imshow("Imagen 100% con desplazamiento", image2)
# cv2.imshow("Imagen con rotacion", image3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# hu1 = hu_moments(image)
# hu2 = hu_moments(image2)

# print("Hu Moments de la imagen 1:")
# print(hu1)
# print("\nHu Moments de la imagen 2:")
# print(hu2)