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

## Funcionamiento del sistema

El funcionamiento general de la aplicación es el siguiente:

1. El usuario ingresa al Servicio Web.
2. Carga una fotografía de una hoja de café.
3. La aplicación muestra la imagen seleccionada.
4. La imagen se ajusta al tamaño requerido por el modelo.
5. El modelo de Inteligencia Artificial analiza la fotografía.
6. El sistema muestra la clasificación obtenida.
7. Se muestra el porcentaje de confianza de la predicción.
8. El resultado se envía a la API de Groq.
9. Groq genera orientación técnica relacionada con el resultado.

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

## Arquitectura del sistema

La arquitectura utilizada es:

Usuario  
↓  
Streamlit Community Cloud  
↓  
Carga de imagen  
↓  
Modelo MobileNetV2  
↓  
Predicción de la hoja  
↓  
Porcentaje de confianza  
↓  
Groq API  
↓  
Orientación técnica  
↓  
Resultado mostrado al usuario

---

## Servicios en la nube utilizados

### Google Colab

Se utilizó para:

- Preparación del dataset.
- Procesamiento de imágenes.
- Entrenamiento del modelo.
- Generación del modelo final.

### GitHub

Se utiliza para almacenar y administrar el código fuente y los archivos necesarios para el funcionamiento del proyecto.

### Streamlit Community Cloud

Se utiliza para desplegar públicamente el Servicio Web.

### Groq API

Se utiliza para generar la orientación técnica y recomendaciones relacionadas con la predicción realizada por el modelo.

---

## Tecnologías utilizadas

- Python
- TensorFlow
- MobileNetV2
- NumPy
- Pillow
- Streamlit
- Groq API
- Google Colab
- Git
- GitHub
- Streamlit Community Cloud

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
