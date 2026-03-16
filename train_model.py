import pickle
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB
from preprocess import prepare_data
from logger_utils import (
    append_training_history,
    log_event,
    log_error,
    save_hyperparameters,
    save_train_io,
    update_model_status,
    create_checkpoint
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"
RESPONSES_PATH = BASE_DIR / "responses.pkl"


def train_and_save():
    try:
        append_training_history("Inicio del entrenamiento del modelo.")
        log_event("Cargando y preparando datos.")

        data = prepare_data(test_size=0.3, random_state=42)

        hyperparameters = {
            "model": "MultinomialNB",
            "test_size": 0.3,
            "random_state": 42,
            "vectorizer": {
                "type": "TfidfVectorizer",
                "lowercase": True
            }
        }
        save_hyperparameters(hyperparameters)

        train_io = {
            "input": {
                "train_samples": len(data["X_train_texts"]),
                "test_samples": len(data["X_test_texts"]),
                "labels": sorted(set(data["y_train"] + data["y_test"]))
            }
        }
        save_train_io(train_io)

        log_event("Entrenando modelo MultinomialNB.")
        model = MultinomialNB()
        model.fit(data["X_train"], data["y_train"])

        with open(MODEL_PATH, "wb") as file:
            pickle.dump(model, file)

        with open(VECTORIZER_PATH, "wb") as file:
            pickle.dump(data["vectorizer"], file)

        with open(RESPONSES_PATH, "wb") as file:
            pickle.dump(data["responses"], file)

        log_event("Modelo, vectorizador y respuestas guardados.")

        checkpoint_model = create_checkpoint(MODEL_PATH, "model_checkpoint.pkl")
        checkpoint_vectorizer = create_checkpoint(VECTORIZER_PATH, "vectorizer_checkpoint.pkl")

        train_io["output"] = {
            "model_file": str(MODEL_PATH.name),
            "vectorizer_file": str(VECTORIZER_PATH.name),
            "responses_file": str(RESPONSES_PATH.name),
            "checkpoint_model": str(checkpoint_model.name),
            "checkpoint_vectorizer": str(checkpoint_vectorizer.name)
        }
        save_train_io(train_io)

        update_model_status(
            status="trained",
            model_path=MODEL_PATH,
            checkpoint_available=True
        )

        append_training_history("Entrenamiento completado correctamente.")
        print("Modelo entrenado y guardado correctamente.")

    except Exception as e:
        log_error(f"Error durante el entrenamiento: {str(e)}")
        update_model_status(status="error", model_path=MODEL_PATH, checkpoint_available=False)
        print("Ocurrió un error durante el entrenamiento. Revisa logs/errors.log")


if __name__ == "__main__":
    train_and_save()