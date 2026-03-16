import json
import re


# =========================
# 1. Limpieza de texto
# =========================
def limpiar_texto(texto):
    """
    Normaliza el texto para el modelo de NLP
    """
    texto = texto.lower()
    texto = texto.strip()
    texto = re.sub(r"[^\w\s]", "", texto)  # elimina puntuación
    return texto


# =========================
# 2. Cargar dataset de intents
# =========================
def cargar_dataset(ruta="dataset.json"):
    """
    Convierte el dataset de intents en pares (texto, etiqueta)
    """
    with open(ruta, "r", encoding="utf-8") as archivo:
        datos = json.load(archivo)

    textos = []
    etiquetas = []

    for intent in datos["intents"]:
        tag = intent["tag"]

        for patron in intent["patterns"]:
            texto_limpio = limpiar_texto(patron)
            textos.append(texto_limpio)
            etiquetas.append(tag)

    return textos, etiquetas


# =========================
# 3. Obtener respuestas por intención
# =========================
def cargar_respuestas(ruta="dataset.json"):
    """
    Crea un diccionario de respuestas por intención
    """
    with open(ruta, "r", encoding="utf-8") as archivo:
        datos = json.load(archivo)

    respuestas = {}

    for intent in datos["intents"]:
        respuestas[intent["tag"]] = intent["responses"]

    return respuestas


# =========================
# 4. Preprocesar texto del usuario
# =========================
def preprocesar_mensaje(mensaje):
    """
    Limpia el mensaje del usuario antes de enviarlo al modelo
    """
    return limpiar_texto(mensaje)