import pickle
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB
from preprocess import prepare_data


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models/model.pkl"
VECTORIZER_PATH = BASE_DIR / "models/vectorizer.pkl"
RESPONSES_PATH = BASE_DIR / "models/responses.pkl"


def train_and_save():
    data = prepare_data()

    model = MultinomialNB()
    model.fit(data["X_train"], data["y_train"])

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    with open(VECTORIZER_PATH, "wb") as file:
        pickle.dump(data["vectorizer"], file)

    with open(RESPONSES_PATH, "wb") as file:
        pickle.dump(data["responses"], file)

    print("Modelo entrenado y guardado correctamente.")
    print(f"Archivo modelo: {MODEL_PATH.name}")
    print(f"Archivo vectorizador: {VECTORIZER_PATH.name}")
    print(f"Archivo respuestas: {RESPONSES_PATH.name}")


if __name__ == "__main__":
    train_and_save()