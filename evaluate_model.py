import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from preprocess import prepare_data
from train_model import MODEL_PATH, VECTORIZER_PATH
from logger_utils import (
    log_event,
    log_error,
    save_validation_results
)


def load_artifacts():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(VECTORIZER_PATH, "rb") as file:
        vectorizer = pickle.load(file)

    return model, vectorizer


def evaluate_model():
    try:
        log_event("Inicio de evaluación del modelo.")

        data = prepare_data(test_size=0.3, random_state=42)
        model, vectorizer = load_artifacts()

        X_test = vectorizer.transform(data["X_test_texts"])
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(data["y_test"], y_pred)
        labels = sorted(set(data["y_test"]))
        matrix = confusion_matrix(data["y_test"], y_pred, labels=labels)
        report = classification_report(data["y_test"], y_pred, output_dict=True)

        examples = []
        errors = []

        for text, real, pred in zip(data["X_test_texts"], data["y_test"], y_pred):
            item = {
                "text": text,
                "real": real,
                "predicted": pred
            }
            examples.append(item)

            if real != pred:
                errors.append(item)

        validation_data = {
            "accuracy": accuracy,
            "labels": labels,
            "confusion_matrix": matrix.tolist(),
            "classification_report": report,
            "examples": examples,
            "errors": errors
        }

        save_validation_results(validation_data)
        log_event("Evaluación completada y resultados guardados.")

        print("===== EVALUACIÓN DEL MODELO =====")
        print(f"Accuracy del modelo: {accuracy:.2f}\n")
        print("Etiquetas:", labels)
        print("\nMatriz de confusión:")
        print(matrix)
        print("\nErrores detectados:", len(errors))

    except Exception as e:
        log_error(f"Error durante la evaluación: {str(e)}")
        print("Ocurrió un error durante la evaluación. Revisa logs/errors.log")


if __name__ == "__main__":
    evaluate_model()