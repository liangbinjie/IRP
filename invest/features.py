import cv2
import numpy as np
from pathlib import Path

CELL_SIZE = 64
BIN_SIZE = 4
N_BINS = CELL_SIZE // BIN_SIZE  # 16 bins por eje


def compute_histogram(img):
    """Histograma horizontal + vertical con ventana de 4 pixeles."""
    norm = img.astype(np.float32) / 255.0

    # Histograma horizontal: suma de pixeles por cada banda de 4 filas
    h_hist = np.array([
        norm[i * BIN_SIZE:(i + 1) * BIN_SIZE, :].sum()
        for i in range(N_BINS)
    ])

    # Histograma vertical: suma de pixeles por cada banda de 4 columnas
    v_hist = np.array([
        norm[:, j * BIN_SIZE:(j + 1) * BIN_SIZE].sum()
        for j in range(N_BINS)
    ])

    # Normalizar por total de pixeles de tinta (cada mitad suma 1)
    total = norm.sum()
    if total > 0:
        h_hist = h_hist / total
        v_hist = v_hist / total

    return np.concatenate([h_hist, v_hist])


def extract_all_features(specimens_dir="img/output"):
    specimens_dir = Path(specimens_dir)
    features = []
    labels = []

    for digit in range(10):
        digit_dir = specimens_dir / str(digit)
        if not digit_dir.exists():
            continue
        imgs = sorted(digit_dir.glob("*.png"))
        for img_path in imgs:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            hist = compute_histogram(img)
            features.append(hist)
            labels.append(digit)

    return np.array(features), np.array(labels)


def save_features(features, labels, path="features.npz"):
    np.savez(path, features=features, labels=labels)
    print(f"  Guardado: {path} ({len(features)} vectores, {features.shape[1]} dimensiones)")


def load_features(path="features.npz"):
    data = np.load(path)
    return data['features'], data['labels']


def _draw_bar_chart(values, title, bar_color, w=380, h=260):
    """Dibuja un grafico de barras usando solo OpenCV. Retorna imagen BGR."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 255

    ml, mr, mt, mb = 52, 15, 35, 38          # margenes left/right/top/bottom
    cw = w - ml - mr                          # ancho del area del grafico
    ch = h - mt - mb                          # alto del area del grafico
    n = len(values)
    max_val = max(values) if max(values) > 0 else 1.0

    # Lineas de grilla horizontales (5 niveles)
    for k in range(5):
        gy = mt + ch - int(k * ch / 4)
        cv2.line(img, (ml, gy), (ml + cw, gy), (210, 210, 210), 1)
        label = f"{max_val * k / 4:.2f}"
        cv2.putText(img, label, (2, gy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (120, 120, 120), 1)

    # Barras
    bar_w = cw // n
    for i, v in enumerate(values):
        bar_h = int((v / max_val) * ch)
        x1 = ml + i * bar_w + 2
        x2 = ml + (i + 1) * bar_w - 2
        y1 = mt + ch - bar_h
        y2 = mt + ch
        cv2.rectangle(img, (x1, y1), (x2, y2), bar_color, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (30, 30, 80), 1)  # borde

    # Ejes
    cv2.line(img, (ml, mt), (ml, mt + ch), (0, 0, 0), 2)
    cv2.line(img, (ml, mt + ch), (ml + cw, mt + ch), (0, 0, 0), 2)

    # Etiquetas X (numeros de banda)
    for i in range(n):
        x = ml + i * bar_w + bar_w // 2 - 5
        cv2.putText(img, str(i), (x, mt + ch + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (60, 60, 60), 1)

    # Titulo del panel
    cv2.putText(img, title, (ml, mt - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)

    # Etiqueta eje X
    cv2.putText(img, "Banda (4px)", (ml + cw // 2 - 35, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 80), 1)

    return img


def visualize_histogram(img, hist, digit, output_path=None):
    h_hist = hist[:N_BINS]
    v_hist = hist[N_BINS:]

    panel_h = 260
    panel_w = 380

    # Panel izquierdo: imagen del digito
    digit_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    digit_panel = np.ones((panel_h, 200, 3), dtype=np.uint8) * 255
    side = min(panel_h - 40, 160)
    resized = cv2.resize(digit_bgr, (side, side))
    ox = (200 - side) // 2
    oy = (panel_h - side) // 2
    digit_panel[oy:oy + side, ox:ox + side] = resized
    cv2.putText(digit_panel, f"Digito: {digit}", (200 // 2 - 35, panel_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Panel central: histograma horizontal (azul)
    h_panel = _draw_bar_chart(h_hist, "Histograma Horizontal (filas)",
                               (180, 110, 40), panel_w, panel_h)

    # Panel derecho: histograma vertical (rojo)
    v_panel = _draw_bar_chart(v_hist, "Histograma Vertical (columnas)",
                               (60, 80, 210), panel_w, panel_h)

    # Titulo general en barra superior
    total_w = 200 + panel_w * 2
    header = np.ones((28, total_w, 3), dtype=np.uint8) * 245
    title_text = f"Histograma del digito \"{digit}\" - ventana 4px"
    cv2.putText(header, title_text, (10, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 1)

    body = np.hstack([digit_panel, h_panel, v_panel])
    canvas = np.vstack([header, body])

    if output_path:
        cv2.imwrite(output_path, canvas)
    else:
        cv2.imshow(f"Histograma digito {digit}", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def generate_sample_histograms(specimens_dir="img/output", output_dir="results/histogramas"):
    specimens_dir = Path(specimens_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for digit in range(10):
        digit_dir = specimens_dir / str(digit)
        if not digit_dir.exists():
            continue
        imgs = sorted(digit_dir.glob("*.png"))
        if not imgs:
            continue

        # Tomar especimen del medio como muestra representativa
        sample_path = imgs[len(imgs) // 2]
        img = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
        hist = compute_histogram(img)

        out_path = output_dir / f"histograma_digito_{digit}.png"
        visualize_histogram(img, hist, digit, str(out_path))
        print(f"  Histograma digito {digit}: {out_path.name}")


if __name__ == "__main__":
    print("Extrayendo caracteristicas...")
    features, labels = extract_all_features()

    if len(features) == 0:
        print("No se encontraron especimenes. Ejecute extract.py primero.")
        exit(1)

    save_features(features, labels)

    print("\nGenerando histogramas de muestra...")
    generate_sample_histograms()

    print(f"\nResumen:")
    print(f"  Total especimenes: {len(features)}")
    print(f"  Dimensiones del vector: {features.shape[1]}")
    for d in range(10):
        print(f"  Digito {d}: {np.sum(labels == d)} especimenes")
