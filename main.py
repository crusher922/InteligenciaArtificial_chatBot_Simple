import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# =========================
# 1. Datos de entrenamiento
# =========================
datos_entrenamiento = [
    ("hola", "saludo"),
    ("buenos dias", "saludo"),
    ("buenas tardes", "saludo"),
    ("que tal", "saludo"),
    ("hola chatbot", "saludo"),

    ("adios", "despedida"),
    ("hasta luego", "despedida"),
    ("nos vemos", "despedida"),
    ("chao", "despedida"),
    ("bye", "despedida"),

    ("cual es el horario", "horario"),
    ("a que hora atienden", "horario"),
    ("cuando abren", "horario"),
    ("cuando cierran", "horario"),
    ("horario de atencion", "horario"),

    ("que cursos ofrecen", "cursos"),
    ("tienen cursos de python", "cursos"),
    ("que puedo estudiar", "cursos"),
    ("ofrecen capacitaciones", "cursos"),
    ("hay cursos disponibles", "cursos"),

    ("como los contacto", "contacto"),
    ("cual es su correo", "contacto"),
    ("numero de telefono", "contacto"),
    ("donde estan ubicados", "contacto"),
    ("como puedo comunicarme", "contacto")
]

textos = [texto for texto, intencion in datos_entrenamiento]
etiquetas = [intencion for texto, intencion in datos_entrenamiento]

# =========================
# 2. Vectorización del texto
# =========================
vectorizador = TfidfVectorizer()
X = vectorizador.fit_transform(textos)

# =========================
# 3. Entrenamiento del modelo
# =========================
modelo = MultinomialNB()
modelo.fit(X, etiquetas)

# =========================
# 4. Respuestas por intención
# =========================
respuestas = {
    "saludo": [
        "Hola, ¿en qué puedo ayudarte?",
        "¡Hola! Estoy aquí para ayudarte.",
        "Buenas, dime tu consulta."
    ],
    "despedida": [
        "Hasta luego.",
        "¡Adiós! Que tengas un buen día.",
        "Nos vemos pronto."
    ],
    "horario": [
        "Nuestro horario de atención es de lunes a viernes de 8:00 a 17:00.",
        "Atendemos de lunes a viernes, de 8 de la mañana a 5 de la tarde."
    ],
    "cursos": [
        "Ofrecemos cursos de Python, desarrollo web y bases de datos.",
        "Tenemos capacitaciones en programación, diseño web y análisis de datos."
    ],
    "contacto": [
        "Puedes contactarnos al correo: info@ejemplo.com",
        "Nuestro número de contacto es 0999999999."
    ]
}

# =========================
# 5. Función del chatbot
# =========================
def responder(mensaje):
    mensaje_vectorizado = vectorizador.transform([mensaje])
    intencion_predicha = modelo.predict(mensaje_vectorizado)[0]
    return random.choice(respuestas[intencion_predicha])

# =========================
# 6. Conversación interactiva
# =========================
print("Chatbot iniciado. Escribe 'salir' para terminar.\n")

while True:
    usuario = input("Tú: ").lower().strip()

    if usuario == "salir":
        print("Bot: Hasta luego.")
        break

    respuesta = responder(usuario)
    print("Bot:", respuesta)