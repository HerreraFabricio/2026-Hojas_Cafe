# Detección de Enfermedades en Hojas de Café

## Descripción

Este proyecto consiste en el desarrollo de un Servicio Web basado en principios de Computación en la Nube que permite detectar enfermedades presentes en hojas de café mediante un modelo de Inteligencia Artificial.

El usuario puede cargar una fotografía de una hoja de café y el sistema realiza una predicción utilizando un modelo de visión artificial entrenado previamente.

Además, se integra la API de Groq para generar orientación técnica relacionada con el resultado obtenido.

---

## Objetivo

Desarrollar una aplicación web que permita detectar enfermedades en hojas de café mediante Inteligencia Artificial y proporcionar recomendaciones técnicas para su manejo preventivo.

---

## Clases detectadas

El modelo fue entrenado para clasificar las hojas de café en cuatro categorías:

- Minador
- Phoma
- Roya
- Saludable

---

## Modelo de Inteligencia Artificial

Para la clasificación de las imágenes se utilizó TensorFlow y MobileNetV2 mediante Transfer Learning.

El modelo fue entrenado en Google Colab utilizando un dataset de imágenes de hojas de café.

El archivo generado por el entrenamiento es:

`mejor_modelo_cafe.keras`

Los nombres y el orden de las clases utilizadas por el modelo se encuentran almacenados en:

`class_names.json`

---


## Orientación generada con Groq

La API de Groq se utiliza para generar automáticamente:

- Descripción del resultado detectado.
- Recomendaciones para el manejo preventivo.
- Buenas prácticas para el cuidado del cultivo.
- Acciones de seguimiento y monitoreo.

La API Key de Groq no se almacena directamente en el código fuente.

En Streamlit Community Cloud se configura mediante Secrets:

`GROQ_API_KEY`

---

## Estructura del proyecto

```text
deteccion-cafe/
│
├── app.py
├── mejor_modelo_cafe.keras
├── class_names.json
├── requirements.txt
└── README.md
