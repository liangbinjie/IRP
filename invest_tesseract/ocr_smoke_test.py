import argparse
import os
import cv2

from features import ocr_digit_from_image


def main():
    parser = argparse.ArgumentParser(description="OCR rapido con Tesseract sobre una imagen.")
    parser.add_argument("image", help="Ruta a la imagen (64x64 o similar)")
    parser.add_argument("--tesseract-cmd", default=os.environ.get("TESSERACT_CMD"))
    args = parser.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("No se pudo cargar la imagen.")
        raise SystemExit(1)

    pred = ocr_digit_from_image(img, tesseract_cmd=args.tesseract_cmd)
    print(f"Prediccion: {pred}")


if __name__ == "__main__":
    main()
