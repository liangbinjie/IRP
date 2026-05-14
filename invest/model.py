import numpy as np
from pathlib import Path


def stratified_split(features, labels, test_ratio=0.30, seed=42):
    """Division estratificada train/test sin sklearn."""
    rng = np.random.RandomState(seed)
    train_idx = []
    test_idx = []

    for digit in range(10):
        idx = np.where(labels == digit)[0].copy()
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * test_ratio))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    return (features[train_idx], features[test_idx],
            labels[train_idx], labels[test_idx])


def train(X_train, y_train):
    """Calcula media y desviacion estandar por clase (modelo estadistico)."""
    model = {}
    for digit in range(10):
        mask = y_train == digit
        if not np.any(mask):
            continue
        class_feats = X_train[mask]
        mean = np.mean(class_feats, axis=0)
        std = np.std(class_feats, axis=0)
        std[std < 1e-8] = 1e-8  # evitar division por cero
        model[digit] = (mean, std)
    return model


def predict(feature, model):
    """Reconoce el digito usando distancia de Mahalanobis minima."""
    best_digit = -1
    best_score = float('inf')

    for digit, (mean, std) in model.items():
        score = np.sum(((feature - mean) / std) ** 2)
        if score < best_score:
            best_score = score
            best_digit = digit

    return best_digit


def predict_batch(features, model):
    return np.array([predict(f, model) for f in features])


def save_model(model, path="model.npz"):
    digits = np.array(sorted(model.keys()))
    means = np.array([model[d][0] for d in digits])
    stds = np.array([model[d][1] for d in digits])
    np.savez(path, digits=digits, means=means, stds=stds)
    print(f"  Modelo guardado: {path} (digitos: {list(digits)})")


def load_model(path="model.npz"):
    data = np.load(path)
    digits = data['digits']
    return {int(d): (data['means'][i], data['stds'][i]) for i, d in enumerate(digits)}


def print_model_summary(model):
    print("\n  Digito | Media promedio | Std promedio")
    print("  " + "-" * 40)
    for digit, (mean, std) in model.items():
        print(f"    {digit}    |   {mean.mean():.5f}      |  {std.mean():.5f}")


if __name__ == "__main__":
    from features import load_features

    print("Cargando caracteristicas...")
    features, labels = load_features()

    print("Dividiendo dataset (70% entrenamiento / 30% prueba)...")
    X_train, X_test, y_train, y_test = stratified_split(features, labels)

    print(f"  Entrenamiento: {len(X_train)} especimenes")
    print(f"  Prueba:        {len(X_test)} especimenes")
    for d in range(10):
        n_tr = np.sum(y_train == d)
        n_te = np.sum(y_test == d)
        print(f"    Digito {d}: {n_tr} train / {n_te} test")

    print("\nEntrenando modelo estadistico...")
    model = train(X_train, y_train)
    print_model_summary(model)

    save_model(model)
    np.savez("splits.npz",
             X_train=X_train, y_train=y_train,
             X_test=X_test, y_test=y_test)
    print("  Splits guardados: splits.npz")
