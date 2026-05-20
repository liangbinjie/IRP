import numpy as np

DEFAULT_TESSERACT_CONFIG = "--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789"


def stratified_split(labels, test_ratio=0.30, seed=42):
    """Division estratificada train/test sin sklearn."""
    rng = np.random.RandomState(seed)
    train_idx = []
    test_idx = []

    for digit in range(10):
        idx = np.where(labels == digit)[0].copy()
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * test_ratio))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())

    return np.array(train_idx), np.array(test_idx)


def get_default_model():
    return {"config": DEFAULT_TESSERACT_CONFIG}


def save_model(model, path="model.npz"):
    np.savez(path, config=np.array([model["config"]]))
    print(f"  Configuracion OCR guardada: {path}")


def load_model(path="model.npz"):
    data = np.load(path, allow_pickle=True)
    return {"config": data["config"][0]}


def print_model_summary(model):
    print("\n  Configuracion OCR:")
    print(f"    {model['config']}")


if __name__ == "__main__":
    try:
        from .features import load_ocr_results
    except ImportError:
        from features import load_ocr_results

    print("Cargando resultados OCR...")
    predictions, labels, _paths = load_ocr_results()

    if len(labels) == 0:
        print("No se encontraron resultados. Ejecute features.py primero.")
        raise SystemExit(1)

    print("Dividiendo dataset (70% entrenamiento / 30% prueba)...")
    train_idx, test_idx = stratified_split(labels)

    print(f"  Entrenamiento: {len(train_idx)} especimenes")
    print(f"  Prueba:        {len(test_idx)} especimenes")

    model = get_default_model()
    print_model_summary(model)
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
