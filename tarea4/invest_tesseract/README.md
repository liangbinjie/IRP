# Invest con Tesseract OCR

Esta carpeta replica el flujo de trabajo de invest/, pero usa Tesseract OCR para
reconocer digitos escritos a mano.

## Requisitos previos

1) Instalar Tesseract OCR (Windows)
   - Recomendado: instalador de UB Mannheim.
   - Asegura que tesseract.exe este en el PATH, o define la variable:

PowerShell:

$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"

## Dependencias Python

Instala dependencias desde la raiz del repo:

pip install -r requirements.txt

## Ejecutar el pipeline completo

1) Copia tus imagenes a img/input/ (en esta carpeta)
2) Ejecuta:

python main.py

## Prueba rapida de OCR

python ocr_smoke_test.py .\img\output\0\digit_00000.png

## Configuracion opcional

Puedes ajustar el OCR con la variable TESSERACT_CONFIG, por ejemplo:

$env:TESSERACT_CONFIG = "--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789"

Archivos generados:
- img/output/{0-9}/    (especimenes extraidos)
- ocr_results.npz      (predicciones OCR y etiquetas)
- splits.npz           (division train/test)
- results/confusion_matrix.png
- results/accuracy_chart.png
- results/ocr_samples.txt
