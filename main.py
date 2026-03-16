import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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
# 2. División de datos
# =========================
X_train_textos, X_test_textos, y_train, y_test = train_test_split(
    textos, etiquetas, test_size=0.3, random_state=42
)

# =========================
# 3. Vectorización y entrenamiento
# =========================
vectorizador = TfidfVectorizer()
X_train = vectorizador.fit_transform(X_train_textos)
X_test = vectorizador.transform(X_test_textos)

modelo = MultinomialNB()
modelo.fit(X_train, y_train)

# =========================
# 4. Evaluación del modelo
# =========================
y_pred = modelo.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
matriz = confusion_matrix(y_test, y_pred)
reporte = classification_report(y_test, y_pred)

print("===== EVALUACIÓN DEL MODELO =====")
print(f"Accuracy del modelo: {accuracy:.2f}\n")

print("Matriz de confusión:")
print(matriz)
print()

print("Reporte de clasificación:")
print(reporte)

# =========================
# 5. Ejemplos de predicción
# =========================
print("\n===== EJEMPLOS DE PREDICCIÓN =====")
for texto, real, predicho in zip(X_test_textos, y_test, y_pred):
    print(f"Texto: '{texto}'")
    print(f"Etiqueta real: {real}")
    print(f"Predicción: {predicho}")
    print("-" * 40)

# =========================
# 6. Errores del modelo
# =========================
print("\n===== ERRORES DEL MODELO =====")
hubo_error = False

for texto, real, predicho in zip(X_test_textos, y_test, y_pred):
    if real != predicho:
        hubo_error = True
        print(f"Texto: '{texto}'")
        print(f"Real: {real} | Predicho: {predicho}")
        print("-" * 40)

if not hubo_error:
    print("No hubo errores en este conjunto de prueba.")

# =========================
# 7. Respuestas por intención
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
# 8. Función de respuesta
# =========================
def responder(mensaje):
    mensaje_vectorizado = vectorizador.transform([mensaje])
    intencion_predicha = modelo.predict(mensaje_vectorizado)[0]
    return random.choice(respuestas[intencion_predicha])

# =========================
# 9. Chat interactivo
# =========================
print("\nChatbot iniciado. Escribe 'salir' para terminar.\n")

while True:
    usuario = input("Tú: ").lower().strip()

    if usuario == "salir":
        print("Bot: Hasta luego.")
        break

    respuesta = responder(usuario)
    print("Bot:", respuesta)