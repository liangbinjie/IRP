import cv2
import numpy as np
import os
import sys
import utils.binarize_cells as binarize
import utils.centrar as centrar


# Tamaño aproximado de celdas
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


# ─────────────────────────────────────────────────────────────────────────────
# Identificar label de cada imagen a partir del nombre del archivo
# ─────────────────────────────────────────────────────────────────────────────

def is_odd_filename(path: str) -> bool:
    """Si el nombre del archivo termina en un dígito impar (1, 3, 5, 7, 9), devuelve True"""
    stem   = os.path.splitext(os.path.basename(path))[0]
    digits = [c for c in stem if c.isdigit()]
    if not digits:
        raise ValueError(f"No digit found in filename: {path}")
    return int(digits[-1]) % 2 == 1


def starting_label(path: str) -> int:
    """Devuelve 0 para archivos con nombres impares (dígitos 0-4), 5 para archivos con nombres pares (dígitos 5-9)."""
    return 0 if is_odd_filename(path) else 5


# ─────────────────────────────────────────────────────────────────────────────
# Deteccion
# ─────────────────────────────────────────────────────────────────────────────

def detect_boxes(gray: np.ndarray):
    """
    Returns (group_boxes, cell_boxes) – each a list of (x, y, w, h) tuples
    sorted in reading order (top-to-bottom, left-to-right).
        * group_boxes: bounding boxes of the large grid groups (labels)
        * cell_boxes: bounding boxes of the individual digit cells
    """
    # uso de canny + dilate para encontrar contornos de celdas y grupos
    edges = cv2.Canny(gray, 30, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    group_boxes: list[tuple] = []
    cell_boxes:  list[tuple] = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
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

    # deduplicamos y ordenamos las cajas de grupos y celdas por orden de lectura
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


# ─────────────────────────────────────────────────────────────────────────────
# Funciones auxiliares para nombrado de archivos y manejo de índices
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Extracción de celdas y guardado
# ─────────────────────────────────────────────────────────────────────────────

def extract_cells(img_path: str, output_root: str = "cells") -> int:
    """
    Crop every digit cell from *img_path* and append it to
    *output_root/<label>/cell_NNNN.png*.

    Existing files are never overwritten: the index starts from the highest
    existing cell number + 1 in each folder.

    Returns the total number of saved cells.
    """
    img  = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    group_boxes, cell_boxes = detect_boxes(gray)

    if not group_boxes:
        print("[extract_cells] WARNING: no group boxes detected – "
              "all cells will be saved under a single label.")

    base_label = starting_label(img_path)

    group_tops    = [b[1] for b in group_boxes]
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
        out_dir = os.path.join(output_root, str(label))
        os.makedirs(out_dir, exist_ok=True)

        if label not in label_offsets:
            label_offsets[label] = next_index(out_dir)

        final_index = label_offsets[label] + cell_index

        # Crop → grayscale → binarize → trim margin → center
        crop   = img[y:y+h, x:x+w]
        crop   = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop   = binarize.binarize_cells(crop)
        margin = 2
        crop   = crop[margin:-margin, margin:-margin]
        crop   = centrar.centrar(crop)

        cv2.imwrite(os.path.join(out_dir, f"cell_{final_index:04d}.png"), crop)

        cell_index += 1
        saved      += 1

    print(f"[Extraccion Celdas] {img_path}  →  {saved} celdas  "
          f"(labels {base_label}–{base_label + len(group_boxes) - 1})")
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Debug helper
# ─────────────────────────────────────────────────────────────────────────────

def save_debug_image(img_path: str, out_path: str = "cells_debug.png") -> None:
    """Overlay group boxes (blue) and cell boxes (green) on the image."""
    img  = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    group_boxes, cell_boxes = detect_boxes(gray)

    for (x, y, w, h) in group_boxes:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 80, 0), 4)
        idx = group_boxes.index((x, y, w, h))
        cv2.putText(img, f"G{idx}", (x+4, y+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 80, 0), 2)

    for (x, y, w, h) in cell_boxes:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 200, 0), 2)

    cv2.imwrite(out_path, img)
    print(f"[debug] {len(group_boxes)} group boxes, "
          f"{len(cell_boxes)} cell boxes  →  {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Batch processing
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def process_all(input_dir: str = "img/input",
                output_root: str = "img/output",
                debug_dir: str = "debug") -> None:
    """
    Procesar todas las imágenes en *input_dir* y guardar las celdas extraídas en
    *output_root/<label>/cell_NNNN.png*.
    """
    image_paths = sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        print(f"No hay imagenes en '{input_dir}'")
        return

    os.makedirs(debug_dir, exist_ok=True)

    total_saved = 0
    for img_path in image_paths:
        try:
            saved = extract_cells(img_path, output_root=output_root)
            total_saved += saved

            # stem       = os.path.splitext(os.path.basename(img_path))[0]
            # debug_path = os.path.join(debug_dir, f"{stem}_debug.png")
            # save_debug_image(img_path, out_path=debug_path)

        except Exception as e:
            print(f"Error procesando {img_path}: {e}")

    print(f"\nProcesado {len(image_paths)} imagenes, "
          f"{total_saved} celdas total.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        extract_cells(img_path, output_root="img/output")
        save_debug_image(img_path, out_path="debug/single_debug.png")
    else:
        process_all(input_dir="img/input", output_root="img/output", debug_dir="debug")