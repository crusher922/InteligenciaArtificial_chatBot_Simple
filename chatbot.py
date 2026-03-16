import pickle
import random
from pathlib import Path
from train_model import MODEL_PATH, VECTORIZER_PATH, RESPONSES_PATH
from logger_utils import log_event, log_error


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
    try:
        model, vectorizer, responses = load_artifacts()
        log_event("Chatbot iniciado correctamente.")

        print("======================================")
        print("Asistente Virtual - Escuela de Programación")
        print("======================================")
        print("Bienvenido.")
        print("Soy un agente virtual diseñado para brindar información")
        print("sobre los servicios de nuestra escuela de programación.")
        print("\nEscribe tu pregunta o escribe 'salir' para terminar.\n")

        while True:
            user_input = input("Tú: ").strip()

            if user_input.lower() == "salir":
                log_event("Usuario finalizó la conversación.")
                print("Bot: Hasta luego.")
                break

            intent = predict_intent(user_input, model, vectorizer)
            answer = random.choice(responses[intent])
            log_event(f"Entrada usuario: {user_input} | Intención detectada: {intent}")
            print("Bot:", answer)

    except Exception as e:
        log_error(f"Error en chatbot: {str(e)}")
        print("Ocurrió un error en el chatbot. Revisa logs/errors.log")


if __name__ == "__main__":
    chatbot()