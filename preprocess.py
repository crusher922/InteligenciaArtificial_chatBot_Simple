import re
import json

# =========================
# 1. Limpieza básica de texto
# =========================
def limpiar_texto(texto):
    """
    Normaliza el texto para el modelo de PLN
    """
    texto = texto.lower()                 # minúsculas
    texto = texto.strip()                 # eliminar espacios extremos
    texto = re.sub(r"[^\w\s]", "", texto) # eliminar signos de puntuación
    return texto


# =========================
# 2. Cargar dataset
# =========================
def cargar_dataset(ruta="dataset.json"):
    """
    Carga el dataset desde un archivo JSON
    """
    with open(ruta, "r", encoding="utf-8") as archivo:
        datos = json.load(archivo)

    textos = []
    etiquetas = []

    for item in datos:
        texto_limpio = limpiar_texto(item["texto"])
        textos.append(texto_limpio)
        etiquetas.append(item["intent"])

    return textos, etiquetas


# =========================
# 3. Preprocesar texto para predicción
# =========================
def preprocesar_mensaje(mensaje):
    """
    Preprocesa un mensaje del usuario antes de enviarlo al modelo
    """
    return limpiar_texto(mensaje)