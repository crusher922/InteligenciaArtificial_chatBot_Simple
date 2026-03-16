import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
RESULTS_FILE = LOGS_DIR / "validation_results.json"
PLOTS_DIR = LOGS_DIR / "plots"


def load_results():
    with open(RESULTS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def ensure_plots_dir():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_accuracy(results):
    accuracy = results["accuracy"]

    plt.figure(figsize=(6, 5))
    plt.bar(["Accuracy"], [accuracy])
    plt.ylim(0, 1)
    plt.title("Accuracy del modelo")
    plt.ylabel("Valor")
    plt.savefig(PLOTS_DIR / "accuracy.png", bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(results):
    matrix = np.array(results["confusion_matrix"])
    labels = results["labels"]

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, interpolation="nearest")
    plt.title("Matriz de confusión")
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center")

    plt.ylabel("Etiqueta real")
    plt.xlabel("Etiqueta predicha")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", bbox_inches="tight")
    plt.close()


def plot_precision_recall_f1(results):
    report = results["classification_report"]

    labels = []
    precision = []
    recall = []
    f1_score = []

    for label, metrics in report.items():
        if label in ["accuracy", "macro avg", "weighted avg"]:
            continue
        labels.append(label)
        precision.append(metrics["precision"])
        recall.append(metrics["recall"])
        f1_score.append(metrics["f1-score"])

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precision, width, label="Precision")
    plt.bar(x, recall, width, label="Recall")
    plt.bar(x + width, f1_score, width, label="F1-score")

    plt.xticks(x, labels, rotation=45)
    plt.ylim(0, 1)
    plt.title("Precision, Recall y F1-score por clase")
    plt.ylabel("Valor")
    plt.xlabel("Clases")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "metrics_by_class.png", bbox_inches="tight")
    plt.close()


def plot_prediction_summary(results):
    total_examples = len(results["examples"])
    total_errors = len(results["errors"])
    total_correct = total_examples - total_errors

    plt.figure(figsize=(6, 5))
    plt.bar(["Correctas", "Errores"], [total_correct, total_errors])
    plt.title("Resumen de predicciones")
    plt.ylabel("Cantidad")
    plt.savefig(PLOTS_DIR / "prediction_summary.png", bbox_inches="tight")
    plt.close()


def main():
    ensure_plots_dir()
    results = load_results()

    plot_accuracy(results)
    plot_confusion_matrix(results)
    plot_precision_recall_f1(results)
    plot_prediction_summary(results)

    print("Gráficas generadas correctamente en:")
    print(PLOTS_DIR)


if __name__ == "__main__":
    main()