# Chatbot Simple con Procesamiento de Lenguaje Natural (PLN)

## Descripción

Este proyecto implementa un **chatbot simple utilizando Procesamiento de Lenguaje Natural (PLN) en Python**.  
El objetivo es demostrar cómo un modelo de Machine Learning puede **interpretar texto y clasificar la intención del usuario** para generar respuestas automáticas.

El sistema analiza el texto ingresado por el usuario, aplica técnicas de **preprocesamiento de lenguaje natural**, y utiliza un **modelo de clasificación** para identificar la intención del mensaje.

Este proyecto fue desarrollado como parte de la asignatura de **Inteligencia Artificial**, con el objetivo de aplicar conceptos básicos de PLN y evaluar el rendimiento del modelo.

---

# Objetivos del proyecto

- Implementar un chatbot simple usando PLN.
- Clasificar la intención del usuario a partir del texto ingresado.
- Analizar el rendimiento del modelo mediante métricas de evaluación.
- Identificar errores del modelo y proponer mejoras.

---

# Tecnologías utilizadas

- Python 3
- Scikit-learn
- NLTK
- NumPy
- JSON
- Matplotlib

---

# Arquitectura del sistema

El funcionamiento del chatbot sigue el siguiente flujo:

Datos → Preprocesamiento → Vectorización → Modelo → Evaluación → Motor de respuestas → Interfaz chatbot


### Descripción del flujo

**Datos**  
El sistema utiliza un dataset en formato JSON que contiene diferentes **intenciones**, ejemplos de frases del usuario y posibles respuestas del chatbot.

**Preprocesamiento**  
El texto se limpia y prepara para ser utilizado por el modelo de Machine Learning.

**Vectorización**  
El texto se transforma en valores numéricos utilizando técnicas como **TF-IDF**, lo que permite que el modelo pueda procesarlo.

**Modelo**  
Se entrena un modelo de clasificación que aprende a identificar la intención del usuario.

**Evaluación**  
El modelo se evalúa utilizando métricas como:

- Accuracy
- Matriz de confusión
- Precision
- Recall
- F1-score

**Motor de respuestas**  
Una vez identificada la intención del usuario, el chatbot selecciona una respuesta correspondiente a esa intención.

**Interfaz chatbot**  
El usuario interactúa con el sistema mediante la consola.

---

# Requisitos del sistema

Se recomienda utilizar:

Python 3.10,3.11 o 3.12

También se recomienda usar un **entorno virtual** para aislar las dependencias del proyecto.

---

# Instalación del proyecto

## 1 Clonar o descargar el proyecto

```bash
git clone https://github.com/usuario/chatbot_pln
cd chatbot_pln

O descargar el proyecto manualmente y acceder a la carpeta.
```
## Crear entorno virtual
python -m venv venv

## Activar entorno en Linux/Mac

source venv/bin/activate

## En windows

venv\Scripts\activate

## Instalar dependencias

pip install scikit-learn nltk numpy matplotlib

## 1 Entrenar el modelo
python train_model.py

## Qué hace este script

Carga el dataset del chatbot

Realiza el preprocesamiento del texto

Convierte el texto a vectores numéricos

Entrena el modelo de clasificación

Guarda el modelo entrenado en archivos

Archivos generados:

model.pkl
vectorizer.pkl
responses.pkl

También se actualizan los archivos de registro dentro de:

logs/

## 2 Evaluar el modelo

python evaluate_model.py

## Qué hace este script

Evalúa el rendimiento del modelo entrenado utilizando un conjunto de datos de prueba.

Calcula métricas como:

Accuracy

Matriz de confusión

Precision

Recall

F1-score

Los resultados se guardan en:

logs/validation_results.json

## 3 Generar gráficas
python plot_results.py

## Qué hace este script

Genera visualizaciones de los resultados de evaluación del modelo.

Las gráficas incluyen:

Accuracy del modelo

Matriz de confusión

Métricas por clase

Resumen de predicciones

Las imágenes se guardan en:

logs/plots/

## Ejemplo:

accuracy.png

confusion_matrix.png

metrics_by_class.png

prediction_summary.png

## 4 Generar dashboard
python generate_dashboard.py

## Qué hace este script

Genera un dashboard en HTML que permite visualizar los resultados del modelo de forma gráfica.

El dashboard incluye:

Métricas del modelo

Estado del modelo

Gráficas de desempeño

Métricas por clase

Errores del modelo

El archivo generado es:

dashboard.html

## 5 Abrir el dashboard

Abrir el archivo en el navegador:

dashboard.html

También se puede abrir desde la terminal.

## Linux

xdg-open dashboard.html

## Windows

start dashboard.html

## 6 Ejecutar el chatbot
python chatbot.py

## Qué hace este script

Permite interactuar con el chatbot desde la consola.

El sistema: recibe el texto del usuario, lo procesa identifica la intención y genera una respuesta adecuada

Ejemplo de conversación:

Tú: hola
Bot: Hola, soy el asistente de la escuela de programación.

Tú: que cursos ofrecen
Bot: Ofrecemos cursos de Python y desarrollo web.

## Para terminar la conversación:

salir

Resultados del modelo

## Los resultados del modelo pueden visualizarse en:

logs/validation_results.json

y en el dashboard generado.