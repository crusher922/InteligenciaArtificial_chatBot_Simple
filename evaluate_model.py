import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from preprocess import prepare_data
from train_model import MODEL_PATH, VECTORIZER_PATH


BASE_DIR = Path(__file__).resolve().parent


def load_artifacts():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(VECTORIZER_PATH, "rb") as file:
        vectorizer = pickle.load(file)

    return model, vectorizer


def evaluate_model():
    data = prepare_data()
    model, vectorizer = load_artifacts()

    X_test = vectorizer.transform(data["X_test_texts"])
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(data["y_test"], y_pred)
    matrix = confusion_matrix(data["y_test"], y_pred, labels=sorted(set(data["y_test"])))
    report = classification_report(data["y_test"], y_pred)

    print("===== EVALUACIÓN DEL MODELO =====")
    print(f"Accuracy del modelo: {accuracy:.2f}\n")

    print("Etiquetas:", sorted(set(data["y_test"])))
    print("\nMatriz de confusión:")
    print(matrix)

    print("\nReporte de clasificación:")
    print(report)

    print("\n===== EJEMPLOS DE PREDICCIÓN =====")
    for text, real, pred in zip(data["X_test_texts"], data["y_test"], y_pred):
        print(f"Texto: '{text}'")
        print(f"Etiqueta real: {real}")
        print(f"Predicción: {pred}")
        print("-" * 40)

    print("\n===== ERRORES DEL MODELO =====")
    has_errors = False
    for text, real, pred in zip(data["X_test_texts"], data["y_test"], y_pred):
        if real != pred:
            has_errors = True
            print(f"Texto: '{text}'")
            print(f"Real: {real} | Predicho: {pred}")
            print("-" * 40)

    if not has_errors:
        print("No hubo errores en este conjunto de prueba.")


if __name__ == "__main__":
    evaluate_model()