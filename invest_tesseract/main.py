from pathlib import Path


def _generate_histograms(specimens_dir, output_dir):
    import importlib.util

    hist_path = Path(__file__).resolve().parents[1] / "invest" / "features.py"
    if not hist_path.exists():
        print("  Aviso: No se encontro invest/features.py, no se generaron histogramas.")
        return

    spec = importlib.util.spec_from_file_location("invest_features", hist_path)
    if spec is None or spec.loader is None:
        print("  Aviso: No se pudo cargar el modulo de histogramas.")
        return

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.generate_sample_histograms(specimens_dir, output_dir)


def run_step(title, fn):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    return fn()


def main():
    Path("results").mkdir(parents=True, exist_ok=True)

    # -- Paso 1: Extraccion de especimenes --
    def step1():
        from extract import extract_all
        counters = extract_all("img/input", "img/output")
        print("\n  Generando histogramas de muestra...")
        _generate_histograms("img/output", "results/histogramas")
        return counters

    run_step("PASO 1 — Extraccion de especimenes", step1)

    # -- Paso 2: OCR con Tesseract --
    def step2():
        from features import run_ocr_on_specimens, save_ocr_results, generate_ocr_samples
        import numpy as np

        predictions, labels, paths = run_ocr_on_specimens("img/output")
        if len(labels) == 0:
            print("  ERROR: No se extrajeron especimenes.")
            return None, None, None

        save_ocr_results(predictions, labels, paths)

        print("\n  Generando muestras de OCR...")
        generate_ocr_samples("img/output", "results/ocr_samples.txt")

        valid = int(np.sum((predictions >= 0) & (predictions <= 9)))
        print(f"\n  Total especimenes: {len(labels)}")
        print(f"  Predicciones validas: {valid}")
        return predictions, labels, paths

    predictions, labels, _paths = run_step("PASO 2 — OCR con Tesseract", step2)
    if predictions is None:
        return

    # -- Paso 3: Division 70/30 (sin entrenamiento) --
    def step3():
        from model import stratified_split, get_default_model, save_model
        import numpy as np

        train_idx, test_idx = stratified_split(labels)

        print(f"  Entrenamiento: {len(train_idx)} especimenes (70%)")
        print(f"  Prueba:        {len(test_idx)} especimenes (30%)")

        model = get_default_model()
        save_model(model)

        np.savez(
            "splits.npz",
            train_idx=train_idx,
            test_idx=test_idx,
            y_train=labels[train_idx],
            y_test=labels[test_idx],
            pred_train=predictions[train_idx],
            pred_test=predictions[test_idx],
        )
        print("  Splits guardados: splits.npz")
        return predictions[test_idx], labels[test_idx]

    pred_test, y_test = run_step("PASO 3 — Division 70/30 (sin entrenamiento)", step3)

    # -- Pasos 4-6: OCR, tasa de aciertos, matriz --
    def step4_6():
        from evaluate import (
            accuracy_per_digit,
            print_accuracy_table,
            compute_confusion_matrix,
            print_confusion_matrix,
            visualize_confusion_matrix,
            visualize_accuracy_chart,
        )

        print("  Evaluando predicciones OCR en prueba...")
        acc_results = accuracy_per_digit(pred_test, y_test)
        print_accuracy_table(acc_results)

        cm = compute_confusion_matrix(pred_test, y_test)
        print_confusion_matrix(cm)

        visualize_confusion_matrix(cm, "results/confusion_matrix.png")
        visualize_accuracy_chart(acc_results, "results/accuracy_chart.png")

    run_step("PASOS 4-6 — OCR, Tasa de Aciertos y Matriz de Confusion", step4_6)

    # -- Resumen final --
    print(f"\n{'='*55}")
    print("  COMPLETADO — Archivos generados:")
    print("    img/output/{0-9}/      Especimenes extraidos")
    print("    results/histogramas/")
    print("    results/ocr_samples.txt")
    print("    results/confusion_matrix.png")
    print("    results/accuracy_chart.png")
    print("    ocr_results.npz        Predicciones OCR y etiquetas")
    print("    model.npz              Configuracion OCR")
    print("    splits.npz             Division train/test")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
