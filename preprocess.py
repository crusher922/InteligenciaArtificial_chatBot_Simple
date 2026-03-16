import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.json"


def load_dataset(dataset_path=DATASET_PATH):
    with open(dataset_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    texts = []
    labels = []
    responses = {}

    for intent in data["intents"]:
        tag = intent["tag"]
        responses[tag] = intent["responses"]

        for pattern in intent["patterns"]:
            texts.append(pattern.lower().strip())
            labels.append(tag)

    return texts, labels, responses


def prepare_data(test_size=0.3, random_state=42):
    texts, labels, responses = load_dataset()

    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    vectorizer = TfidfVectorizer(lowercase=True)
    X_train = vectorizer.fit_transform(X_train_texts)
    X_test = vectorizer.transform(X_test_texts)

    return {
        "X_train_texts": X_train_texts,
        "X_test_texts": X_test_texts,
        "y_train": y_train,
        "y_test": y_test,
        "X_train": X_train,
        "X_test": X_test,
        "vectorizer": vectorizer,
        "responses": responses
    }


if __name__ == "__main__":
    data = prepare_data()
    print("Datos de entrenamiento:", len(data["X_train_texts"]))
    print("Datos de prueba:", len(data["X_test_texts"]))
    print("Etiquetas:", sorted(set(data["y_train"] + data["y_test"])))
