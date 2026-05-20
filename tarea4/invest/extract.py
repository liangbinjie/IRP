import cv2
import numpy as np
import re
from pathlib import Path

DIGITS_PER_IMAGE = 5
ROWS_PER_DIGIT = 3
CELL_SIZE = 64


def get_starting_digit(image_path):
    # Usar el ULTIMO grupo de digitos (numero de secuencia: 0001, 0002...)
    # no el primero (que seria la fecha 20260505)
    matches = re.findall(r'\d+', Path(image_path).stem)
    if matches:
        num = int(matches[-1])
        return 0 if num % 2 == 1 else 5
    return 0


def binarize(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def detect_grid_lines(binary):
    h, w = binary.shape

    # Horizontal: lineas que abarcan al menos 1/8 del ancho
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 8, 1))
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)

    # Vertical: lineas que abarcan al menos 1/50 del alto (una celda)
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 50))
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vk)

    grid = cv2.bitwise_or(horiz, vert)
    # Dilatar para cerrar gaps en intersecciones
    grid = cv2.dilate(grid, np.ones((3, 3), np.uint8), iterations=2)
    return grid


def find_cell_holes(grid, img_shape):
    h, w = img_shape

    # RETR_CCOMP: nivel 0 = estructura, nivel 1 = huecos (celdas)
    contours, hierarchy = cv2.findContours(grid, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []

    cells = []
    min_side = min(h, w) * 0.02
    max_side = min(h, w) * 0.12

    for i, c in enumerate(contours):
        # Solo huecos (tienen padre en la jerarquía)
        if hierarchy[0][i][3] == -1:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < min_side or ch < min_side:
            continue
        if cw > max_side or ch > max_side:
            continue
        if not (0.4 < cw / ch < 2.5):
            continue
        cells.append((x, y, cw, ch))

    return cells


def cluster_into_rows(cells, tolerance=15):
    if not cells:
        return []

    items = sorted([(c[1] + c[3] // 2, c) for c in cells])
    rows = []
    current_row = [items[0][1]]
    current_y = items[0][0]

    for yc, cell in items[1:]:
        if yc - current_y < tolerance:
            current_row.append(cell)
            current_y = yc
        else:
            rows.append(current_row)
            current_row = [cell]
            current_y = yc
    rows.append(current_row)
    return rows


def label_cells(rows, starting_digit):
    """Asigna digitos usando los 4 gaps mas grandes entre filas como separadores de grupos."""
    if not rows:
        return []

    n_groups = DIGITS_PER_IMAGE  # 5 grupos por imagen

    row_y = np.array([np.mean([c[1] + c[3] // 2 for c in row]) for row in rows])

    if len(rows) >= n_groups + 1:
        gaps = np.diff(row_y)
        # Los 4 gaps mas grandes separan los 5 grupos de digitos
        boundary_rows = sorted(np.argsort(gaps)[::-1][:n_groups - 1])
        group_starts = [0] + [b + 1 for b in boundary_rows]
    else:
        # Fallback: dividir uniformemente
        group_starts = [i * ROWS_PER_DIGIT for i in range(n_groups)]

    labeled = []
    ends = group_starts[1:] + [len(rows)]
    for g_idx, (start, end) in enumerate(zip(group_starts, ends)):
        digit = starting_digit + g_idx
        if digit > 9:
            continue
        for row_idx in range(start, end):
            for cell in sorted(rows[row_idx], key=lambda c: c[0]):
                labeled.append((cell, digit))
    return labeled


def preprocess_cell(binary, x, y, w, h, size=CELL_SIZE):
    margin = max(3, min(w, h) // 10)
    x1 = max(0, x + margin)
    y1 = max(0, y + margin)
    x2 = min(binary.shape[1], x + w - margin)
    y2 = min(binary.shape[0], y + h - margin)

    if x2 <= x1 or y2 <= y1:
        return None

    cell = binary[y1:y2, x1:x2].copy()

    # Limpiar ruido con apertura morfologica
    cell = cv2.morphologyEx(cell, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    pixels = cv2.findNonZero(cell)
    if pixels is None or len(pixels) < 15:
        return None

    dx, dy, dw, dh = cv2.boundingRect(pixels)
    if dw < 5 or dh < 5:
        return None

    digit_crop = cell[dy:dy + dh, dx:dx + dw]

    # Centrar en cuadrado
    side = max(dw, dh)
    pad_top = (side - dh) // 2
    pad_bot = side - dh - pad_top
    pad_left = (side - dw) // 2
    pad_right = side - dw - pad_left

    padded = cv2.copyMakeBorder(digit_crop, pad_top, pad_bot, pad_left, pad_right,
                                 cv2.BORDER_CONSTANT, value=0)
    # Margen extra
    extra = size // 8
    padded = cv2.copyMakeBorder(padded, extra, extra, extra, extra,
                                  cv2.BORDER_CONSTANT, value=0)

    resized = cv2.resize(padded, (size, size), interpolation=cv2.INTER_AREA)
    _, final = cv2.threshold(resized, 50, 255, cv2.THRESH_BINARY)
    return final


def process_image(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  No se pudo cargar: {image_path}")
        return {}

    starting_digit = get_starting_digit(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = binarize(gray)
    grid = detect_grid_lines(binary)
    cells = find_cell_holes(grid, binary.shape)

    if not cells:
        print(f"  No se encontraron celdas en: {Path(image_path).name}")
        return {}

    rows = cluster_into_rows(cells)
    labeled = label_cells(rows, starting_digit)

    specimens = {}
    for (x, y, w, h), digit in labeled:
        cell_img = preprocess_cell(binary, x, y, w, h)
        if cell_img is not None:
            specimens.setdefault(digit, []).append(cell_img)

    return specimens


def extract_all(input_dir="img/input", output_dir="img/output"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for d in range(10):
        (output_dir / str(d)).mkdir(parents=True, exist_ok=True)

    counters = {d: 0 for d in range(10)}
    image_files = sorted(input_dir.glob("*.png")) + sorted(input_dir.glob("*.jpg"))

    for img_path in image_files:
        print(f"  Procesando: {img_path.name}")
        specimens = process_image(img_path)

        for digit, cells in specimens.items():
            for cell_img in cells:
                out_path = output_dir / str(digit) / f"digit_{counters[digit]:05d}.png"
                cv2.imwrite(str(out_path), cell_img)
                counters[digit] += 1

    print("\nExtraccion completada:")
    for d in range(10):
        print(f"  Digito {d}: {counters[d]} especimenes")

    return counters


if __name__ == "__main__":
    extract_all()
