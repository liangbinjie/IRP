import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def compute_confusion_matrix(predictions, labels, n_classes=10):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(labels, predictions):
        if 0 <= true < n_classes and 0 <= pred < n_classes:
            cm[true][pred] += 1
    return cm


def accuracy_per_digit(predictions, labels):
    results = {}
    for digit in range(10):
        mask = labels == digit
        total = int(np.sum(mask))
        correct = int(np.sum(predictions[mask] == digit))
        results[digit] = (correct, total, correct / total if total > 0 else 0.0)
    return results


def print_accuracy_table(acc_results):
    print("\n  === Tasa de Aciertos por Digito ===")
    print(f"  {'Digito':<10} {'Correctos':<12} {'Total':<10} {'Tasa':<10}")
    print("  " + "-" * 45)
    total_correct = 0
    total_all = 0
    for digit in range(10):
        correct, total, rate = acc_results[digit]
        total_correct += correct
        total_all += total
        print(f"  {digit:<10} {correct:<12} {total:<10} {rate:.4f}")
    print("  " + "-" * 45)
    overall = total_correct / total_all if total_all > 0 else 0.0
    print(f"  {'TOTAL':<10} {total_correct:<12} {total_all:<10} {overall:.4f}")


def print_confusion_matrix(cm):
    print("\n  === Matriz de Confusion ===")
    print("  (Filas = Real, Columnas = Prediccion)\n")
    header = "       " + "  ".join(f"{i:3d}" for i in range(10))
    print(header)
    print("       " + "-" * 43)
    for i, row in enumerate(cm):
        print(f"   {i} | " + "  ".join(f"{v:3d}" for v in row))


def visualize_confusion_matrix(cm, output_path=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(range(10), fontsize=11)
    ax.set_yticklabels(range(10), fontsize=11)
    ax.set_xlabel('Prediccion', fontsize=13)
    ax.set_ylabel('Real', fontsize=13)
    ax.set_title('Matriz de Confusion — Reconocedor Estadistico', fontsize=14)

    thresh = cm.max() / 2.0
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i][j]),
                    ha='center', va='center', fontsize=9,
                    color='white' if cm[i][j] > thresh else 'black')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Matriz guardada: {output_path}")
    else:
        plt.show()


def visualize_accuracy_chart(acc_results, output_path=None):
    digits = list(range(10))
    rates = [acc_results[d][2] for d in digits]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(digits, rates, color='steelblue', edgecolor='navy', linewidth=0.7)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(digits)
    ax.set_xlabel('Digito', fontsize=12)
    ax.set_ylabel('Tasa de aciertos', fontsize=12)
    ax.set_title('Tasa de Aciertos por Digito', fontsize=14)
    ax.axhline(y=sum(rates) / len(rates), color='red', linestyle='--',
               label=f'Promedio: {sum(rates)/len(rates):.3f}')
    ax.legend()

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{rate:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Grafico de aciertos guardado: {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    from model import load_model, predict_batch

    print("Cargando modelo y datos de prueba...")
    model = load_model()
    splits = np.load("splits.npz")
    X_test = splits['X_test']
    y_test = splits['y_test']

    print(f"  Evaluando {len(X_test)} especimenes de prueba...")
    predictions = predict_batch(X_test, model)

    acc_results = accuracy_per_digit(predictions, y_test)
    print_accuracy_table(acc_results)

    cm = compute_confusion_matrix(predictions, y_test)
    print_confusion_matrix(cm)

    Path("results").mkdir(exist_ok=True)
    visualize_confusion_matrix(cm, "results/confusion_matrix.png")
    visualize_accuracy_chart(acc_results, "results/accuracy_chart.png")
