from pathlib import Path


def run_step(title, fn):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    return fn()


def main():
    Path("results/histogramas").mkdir(parents=True, exist_ok=True)

    # ── Paso 1: Extraccion de especimenes ─────────────────────
    def step1():
        from extract import extract_all
        return extract_all("img/input", "img/output")

    run_step("PASO 1 — Extraccion de especimenes", step1)

    # ── Paso 2: Histogramas ────────────────────────────────────
    def step2():
        from features import extract_all_features, save_features, generate_sample_histograms

        features, labels = extract_all_features("img/output")
        if len(features) == 0:
            print("  ERROR: No se extrajeron especimenes.")
            return None, None

        save_features(features, labels)

        print("\n  Generando histogramas de muestra (uno por digito)...")
        generate_sample_histograms("img/output", "results/histogramas")

        print(f"\n  Total especimenes: {len(features)}")
        print(f"  Dimensiones del vector: {features.shape[1]}")
        return features, labels

    features, labels = run_step("PASO 2 — Extraccion de histogramas", step2)
    if features is None:
        return

    # ── Paso 3: Modelo estadistico (70% entrenamiento) ────────
    def step3():
        from model import stratified_split, train, save_model, print_model_summary
        import numpy as np

        X_train, X_test, y_train, y_test = stratified_split(features, labels)

        print(f"  Entrenamiento: {len(X_train)} especimenes (70%)")
        print(f"  Prueba:        {len(X_test)} especimenes (30%)")

        model = train(X_train, y_train)
        print_model_summary(model)

        save_model(model)
        np.savez("splits.npz",
                 X_train=X_train, y_train=y_train,
                 X_test=X_test, y_test=y_test)
        print("  Splits guardados: splits.npz")
        return model, X_test, y_test

    model, X_test, y_test = run_step("PASO 3 — Modelo estadistico (media + std por digito)", step3)

    # ── Pasos 4-6: Reconocimiento, tasa de aciertos, matriz ───
    def step4_6():
        from model import predict_batch
        from evaluate import (accuracy_per_digit, print_accuracy_table,
                               compute_confusion_matrix, print_confusion_matrix,
                               visualize_confusion_matrix, visualize_accuracy_chart)

        print("  Reconociendo especimenes de prueba...")
        predictions = predict_batch(X_test, model)

        acc_results = accuracy_per_digit(predictions, y_test)
        print_accuracy_table(acc_results)

        cm = compute_confusion_matrix(predictions, y_test)
        print_confusion_matrix(cm)

        visualize_confusion_matrix(cm, "results/confusion_matrix.png")
        visualize_accuracy_chart(acc_results, "results/accuracy_chart.png")

    run_step("PASOS 4-6 — Reconocimiento, Tasa de Aciertos y Matriz de Confusion", step4_6)

    # ── Resumen final ──────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  COMPLETADO — Archivos generados:")
    print("    img/output/{{0-9}}/    Especimenes extraidos")
    print("    results/histogramas/  Histogramas por digito")
    print("    results/confusion_matrix.png")
    print("    results/accuracy_chart.png")
    print("    features.npz          Vectores de caracteristicas")
    print("    model.npz             Modelo estadistico")
    print("    splits.npz            Division train/test")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
