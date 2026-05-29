import cv2
import numpy as np


def create_table(rows, cols, cell_width, cell_height):
    width = cols * cell_width
    height = rows * cell_height

    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Generacion de lineas horizontales
    for r in range(rows + 1):
        y = r * cell_height
        cv2.line(img, (0, y), (width, y), (0, 0, 0), 2)

    # Generacion de lineas verticales
    for c in range(cols + 1):
        x = c * cell_width
        cv2.line(img, (x, 0), (x, height), (0, 0, 0), 2)

    return img

rows = 8
cols = 5

cell_width = 180
cell_height = 180

img = create_table(rows, cols, cell_width, cell_height)

# Carga de imagenes a procesar
digit = 9
img_to_process = cv2.imread(f'img/hu_input/{digit}.png')
from utils.mapeo import mapeo_lineal

images = [
    img_to_process,
    cv2.rotate(img_to_process, cv2.ROTATE_90_CLOCKWISE),
    # mapeo_lineal(img_to_process, a=0.5, b=0),
    cv2.resize(img_to_process, (0, 0), fx=0.5, fy=0.5),
    mapeo_lineal(img_to_process, a=1, b=6)                  # desplazamiento
]

hu_sets = []

# Put images in header row
for i, pic in enumerate(images):

    pic

    if pic is None:
        continue

    # Resize image to fit inside cell
    pic = cv2.resize(
        pic,
        (cell_width - 20, cell_height - 20)
    )

    x = (i + 1) * cell_width + 10
    y = 10

    h, w = pic.shape[:2]

    img[y:y+h, x:x+w] = pic

    import hu
    hu_values = hu.hu_moments(pic).flatten()
    hu_sets.append(hu_values)

# First column labels
for r in range(7):

    y = (r + 1) * cell_height + 50

    cv2.putText(
        img,
        f"Hu {r+1}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2
    )

    # Add Hu values
    for c in range(4):

        value = hu_sets[c][r]

        x = (c + 1) * cell_width + 10

        cv2.putText(
            img,
            f"{value:.6e}",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )

cv2.imshow("Hu Moments Table", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f"img/hu_tables/{digit}.png", img)