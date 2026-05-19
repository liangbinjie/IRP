import os
from pathlib import Path

import cv2
import numpy as np

DEFAULT_TESSERACT_CONFIG = "--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789"


def _resolve_tesseract_cmd(explicit_cmd=None):
    return explicit_cmd or os.environ.get("TESSERACT_CMD")


def _resolve_tesseract_config(explicit_config=None):
    return explicit_config or os.environ.get("TESSERACT_CONFIG") or DEFAULT_TESSERACT_CONFIG


def _load_pytesseract():
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Faltan dependencias: pytesseract y pillow.") from exc
    return pytesseract, Image


def _prepare_for_ocr(img):
    if img is None:
        return None

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invertir si el fondo es negro y el trazo es blanco
    if np.mean(img) < 127:
        img = cv2.bitwise_not(img)

    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img


def _parse_digit_text(text):
    if not text:
        return -1
    digits = [c for c in text if c.isdigit()]
    return int(digits[0]) if digits else -1


def ocr_digit_from_image(img, tesseract_cmd=None, config=None):
    pytesseract, Image = _load_pytesseract()
    cmd = _resolve_tesseract_cmd(tesseract_cmd)
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd

    config = _resolve_tesseract_config(config)
    prepped = _prepare_for_ocr(img)
    if prepped is None:
        return -1

    pil_img = Image.fromarray(prepped)
    text = pytesseract.image_to_string(pil_img, config=config)
    return _parse_digit_text(text)


def run_ocr_on_specimens(specimens_dir="img/output", tesseract_cmd=None, config=None):
    specimens_dir = Path(specimens_dir)
    predictions = []
    labels = []
    paths = []

    for digit in range(10):
        digit_dir = specimens_dir / str(digit)
        if not digit_dir.exists():
            continue
        imgs = sorted(digit_dir.glob("*.png"))
        for img_path in imgs:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            pred = ocr_digit_from_image(img, tesseract_cmd=tesseract_cmd, config=config)
            predictions.append(pred)
            labels.append(digit)
            paths.append(str(img_path))

    return np.array(predictions), np.array(labels), np.array(paths, dtype=object)


def save_ocr_results(predictions, labels, paths, path="ocr_results.npz"):
    np.savez(path, predictions=predictions, labels=labels, paths=paths)
    print(f"  Guardado: {path} ({len(labels)} especimenes)")


def load_ocr_results(path="ocr_results.npz"):
    data = np.load(path, allow_pickle=True)
    return data["predictions"], data["labels"], data["paths"]


def generate_ocr_samples(specimens_dir="img/output", output_path="results/ocr_samples.txt",
                         tesseract_cmd=None, config=None):
    specimens_dir = Path(specimens_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["digit,true,pred,file\n"]
    for digit in range(10):
        digit_dir = specimens_dir / str(digit)
        if not digit_dir.exists():
            continue
        imgs = sorted(digit_dir.glob("*.png"))
        if not imgs:
            continue
        sample_path = imgs[len(imgs) // 2]
        img = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
        pred = ocr_digit_from_image(img, tesseract_cmd=tesseract_cmd, config=config)
        lines.append(f"{digit},{digit},{pred},{sample_path}\n")

    output_path.write_text("".join(lines), encoding="utf-8")
    print(f"  Muestras OCR guardadas: {output_path}")


if __name__ == "__main__":
    print("Ejecutando OCR sobre especimenes...")
    predictions, labels, paths = run_ocr_on_specimens()

    if len(labels) == 0:
        print("No se encontraron especimenes. Ejecute extract.py primero.")
        raise SystemExit(1)

    save_ocr_results(predictions, labels, paths)
    generate_ocr_samples()

    valid = int(np.sum((predictions >= 0) & (predictions <= 9)))
    print("\nResumen:")
    print(f"  Total especimenes: {len(labels)}")
    print(f"  Predicciones validas: {valid}")
    for d in range(10):
        print(f"  Digito {d}: {np.sum(labels == d)} especimenes")
