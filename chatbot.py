import pickle
import random
from pathlib import Path
from train_model import MODEL_PATH, VECTORIZER_PATH, RESPONSES_PATH


def load_artifacts():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(VECTORIZER_PATH, "rb") as file:
        vectorizer = pickle.load(file)

    with open(RESPONSES_PATH, "rb") as file:
        responses = pickle.load(file)

    return model, vectorizer, responses


def predict_intent(message, model, vectorizer):
    vectorized_message = vectorizer.transform([message.lower().strip()])
    return model.predict(vectorized_message)[0]


def chatbot():
    model, vectorizer, responses = load_artifacts()

    print("======================================")
    print("🤖 Asistente virtual de la Escuela de Programación")
    print("======================================")
    print("Hola, soy el asistente virtual de la Escuela de Programación.")
    print("Estoy aquí para brindarte información sobre:")
    print("- Cursos disponibles")
    print("- Horarios de atención")
    print("- Contacto de la escuela")
    print()
    print("Puedes hacerme preguntas como:")
    print("• ¿Qué cursos ofrecen?")
    print("• ¿Cuál es el horario?")
    print("• ¿Cómo puedo contactarlos?")
    print()
    print("Escribe 'salir' para terminar la conversación.\n")

    while True:
        user_input = input("Tú: ").strip()

        if user_input.lower() == "salir":
            print("Bot: Hasta luego.")
            break

        intent = predict_intent(user_input, model, vectorizer)
        answer = random.choice(responses[intent])
        print("Bot:", answer)


if __name__ == "__main__":
    chatbot()