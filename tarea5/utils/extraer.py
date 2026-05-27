
import cv2
import numpy as np
import os
from . import centrar, binarizar

CELL_AREA_MIN    = 4_000
CELL_AREA_MAX    = 18_000
CELL_ASPECT_MIN  = 0.5
CELL_ASPECT_MAX  = 2.0

# Cuadricula grande - tamaño aprox
GROUP_AREA_MIN   = 200_000   
GROUP_ASPECT_MIN = 3.0       
GROUP_ASPECT_MAX = 8.0

DEDUP_THRESHOLD  = 15        # px – ignore duplicate contours closer than this
ROW_SNAP         = 50        # px – bucket size for reading-order sort

def next_index(folder: str) -> int:
    """
    Retorna el siguiente índice disponible para guardar una nueva celda en *folder*,
    asumiendo que los archivos existentes siguen el formato "cell_NNNN.png" con NNNN siendo un número entero.
    Si el folder no existe o no contiene archivos válidos, retorna 0.
    """
    if not os.path.isdir(folder):
        return 0
    indices = []
    for fname in os.listdir(folder):
        stem, ext = os.path.splitext(fname)
        if ext.lower() == ".png" and stem.startswith("cell_"):
            try:
                indices.append(int(stem.split("_")[1]))
            except (IndexError, ValueError):
                pass
    return max(indices) + 1 if indices else 0

def detectar_celdas(gray_img):
    edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
    dilate = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=3)
    contours, _ = cv2.findContours(dilate, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    group_boxes = []
    cell_boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area   = w * h
        aspect = w / h if h > 0 else 0

        # si es un grupo grande, lo guardamos como caja de grupo; si es una celda pequeña, la guardamos como caja de celda
        if (area >= GROUP_AREA_MIN
                and GROUP_ASPECT_MIN <= aspect <= GROUP_ASPECT_MAX):
            group_boxes.append((x, y, w, h))

        # si es una celda, la guardamos como caja de celda; el resto se ignora (ruido, líneas, etc.)
        elif (CELL_AREA_MIN < area < CELL_AREA_MAX
              and CELL_ASPECT_MIN < aspect < CELL_ASPECT_MAX
              and w > 10 and h > 10):
            cell_boxes.append((x, y, w, h))

    # Deduplicamos y ordenamos las cajas
    def dedup(boxes):
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        unique = []
        for bx, by, bw, bh in boxes:
            if not any(abs(bx - ux) < DEDUP_THRESHOLD
                       and abs(by - uy) < DEDUP_THRESHOLD
                       for ux, uy, *_ in unique):
                unique.append((bx, by, bw, bh))
        return unique

    def reading_order(boxes):
        return sorted(boxes, key=lambda b: (round(b[1] / ROW_SNAP) * ROW_SNAP, b[0]))

    return reading_order(dedup(group_boxes)), reading_order(dedup(cell_boxes))

def extraer_celdas(img, output_dir, isOdd) -> int:
    """
    Dada una imagen, extraer las celdas y guardarlas en un directorio de salida.
    Devuelve el número de celdas guardadas.

    Si el parametro *idOdd* es verdadero, los digitos de etiqueta son [0,1,2,3,4]
    Si el parametro *idOdd* es falso, los digitos de etiqueta son [5,6,7,8,9]
    """

    if img is None:
        raise ValueError("La imagen proporcionada es None. Asegúrese de que la ruta sea correcta y el archivo exista.")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    group_boxes, cell_boxes = detectar_celdas(gray)

    if not group_boxes:
        print(f"ADVERTENCIA: No se detectaron grupos en esta imagen.")
    
    base_label = 0 if isOdd else 5

    group_tops = [b[1] for b in group_boxes]
    group_heights = [b[3] for b in group_boxes]

    def group_index_for(cy: int) -> int:
        """Return which group box (0-based) contains a cell at vertical pos cy."""
        for i, (top, height) in enumerate(zip(group_tops, group_heights)):
            if top <= cy < top + height:
                return i
        if not group_tops:
            return 0
        dists = [abs(cy - (top + height // 2))
                 for top, height in zip(group_tops, group_heights)]
        return int(np.argmin(dists))

    # Pre-compute the starting index for each label folder once,
    # so we don't re-scan the directory on every cell write.
    label_offsets: dict[int, int] = {}

    saved      = 0
    prev_group = None
    label      = base_label
    cell_index = 0          # count of cells written to current label this run

    for (x, y, w, h) in cell_boxes:
        g = group_index_for(y)

        if prev_group is None:
            prev_group = g

        if g != prev_group:
            label      = base_label + g
            cell_index = 0
            prev_group = g

        # Resolve output folder and its append offset (computed once per label)
        out_dir = os.path.join(output_dir, str(label))
        os.makedirs(out_dir, exist_ok=True)
    
        if label not in label_offsets:
            label_offsets[label] = next_index(out_dir)

        final_index = label_offsets[label] + cell_index

        crop   = img[y:y+h, x:x+w]
        crop   = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop   = binarizar.binarizar(crop)
        margin = 2
        crop   = crop[margin:-margin, margin:-margin]
        crop   = centrar.centrar(crop)

        cv2.imwrite(os.path.join(out_dir, f"cell_{final_index:04d}.png"), crop)

        cell_index += 1
        saved      += 1

    return saved