import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from groq import Groq


# ---------------------------------------------------------
# CONFIGURACION GENERAL
# ---------------------------------------------------------

st.set_page_config(
    page_title="Detección de Enfermedades en Café",
    page_icon="🌿",
    layout="wide"
)


# ---------------------------------------------------------
# ESTILOS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .titulo {
        font-size: 42px;
        font-weight: 800;
        color: #244b2f;
        margin-bottom: 5px;
    }

    .subtitulo {
        font-size: 17px;
        color: #66756b;
        margin-bottom: 30px;
    }

    .resultado {
        background-color: #f4f7f1;
        border: 1px solid #dce5d8;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 20px;
    }

    .enfermedad {
        font-size: 30px;
        font-weight: 800;
        color: #263d2a;
    }

    .confianza {
        font-size: 38px;
        font-weight: 800;
        color: #7b4b20;
    }

    .orientacion {
        background-color: #f7f4ec;
        border: 1px solid #e6dfce;
        border-radius: 16px;
        padding: 25px;
        margin-top: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# CARGAR MODELO
# ---------------------------------------------------------

@st.cache_resource
def cargar_modelo():
    modelo = tf.keras.models.load_model(
        "mejor_modelo_cafe.keras"
    )

    return modelo


# ---------------------------------------------------------
# CARGAR NOMBRES DE CLASES
# ---------------------------------------------------------

@st.cache_data
def cargar_clases():
    with open(
        "class_names.json",
        "r",
        encoding="utf-8"
    ) as archivo:

        clases = json.load(archivo)

    return clases


modelo = cargar_modelo()
class_names = cargar_clases()


# ---------------------------------------------------------
# PREPARAR IMAGEN
# ---------------------------------------------------------

def preparar_imagen(imagen):

    imagen = imagen.convert("RGB")

    imagen = imagen.resize(
        (224, 224)
    )

    arreglo = np.array(
        imagen,
        dtype=np.float32
    )

    arreglo = np.expand_dims(
        arreglo,
        axis=0
    )

    return arreglo


# ---------------------------------------------------------
# REALIZAR PREDICCION
# ---------------------------------------------------------

def predecir(imagen):

    imagen_preparada = preparar_imagen(
        imagen
    )

    prediccion = modelo.predict(
        imagen_preparada,
        verbose=0
    )

    indice = int(
        np.argmax(prediccion[0])
    )

    confianza = float(
        prediccion[0][indice]
    )

    enfermedad = class_names[indice]

    return enfermedad, confianza


# ---------------------------------------------------------
# OBTENER API KEY DE GROQ
# ---------------------------------------------------------

def obtener_groq_api_key():

    try:
        return st.secrets["GROQ_API_KEY"]

    except Exception:
        return os.getenv("GROQ_API_KEY")


# ---------------------------------------------------------
# GENERAR ORIENTACION CON GROQ
# ---------------------------------------------------------

def generar_orientacion(
    enfermedad,
    confianza
):

    api_key = obtener_groq_api_key()

    if not api_key:
        return None

    cliente = Groq(
        api_key=api_key
    )

    prompt = f"""
    Actúa como asistente técnico especializado en el cultivo de café.

    Un modelo de inteligencia artificial analizó una fotografía
    de una hoja de café.

    Resultado detectado:
    {enfermedad}

    Confianza del modelo:
    {confianza:.2f}%

    Genera una orientación clara y sencilla para un productor de café.

    Debes incluir exactamente estas secciones:

    1. Descripción
    Explica brevemente qué significa el resultado detectado.

    2. Manejo preventivo
    Proporciona recomendaciones preventivas y de manejo.

    3. Buenas prácticas
    Indica buenas prácticas para el cuidado del cultivo.

    4. Seguimiento y monitoreo
    Explica qué debe observar el productor en los siguientes días.

    Importante:

    - No afirmes que el resultado sustituye un diagnóstico profesional.
    - Indica que la predicción proviene de un modelo de inteligencia artificial.
    - Si el resultado es Saludable, da recomendaciones para mantener
      la planta en buenas condiciones.
    - No inventes productos químicos específicos ni dosis.
    - Recomienda consultar a un técnico agrícola cuando los síntomas
      sean graves o continúen avanzando.
    """

    respuesta = cliente.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un asistente técnico especializado "
                    "en orientación básica sobre cultivos de café."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.4,
        max_tokens=900
    )

    return respuesta.choices[0].message.content


# ---------------------------------------------------------
# TITULO
# ---------------------------------------------------------

st.markdown(
    """
    <div class="titulo">
        🌿 Detección de Enfermedades en Hojas de Café
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitulo">
        Cargue una fotografía de una hoja de café para analizarla
        mediante inteligencia artificial.
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# INTERFAZ
# ---------------------------------------------------------

columna_imagen, columna_resultado = st.columns(
    [1.15, 1]
)


# ---------------------------------------------------------
# COLUMNA IZQUIERDA
# ---------------------------------------------------------

with columna_imagen:

    st.subheader(
        "📷 Captura o carga de imagen"
    )

    archivo = st.file_uploader(
        "Seleccione una fotografía de la hoja",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if archivo is not None:

        imagen = Image.open(
            archivo
        )

        st.image(
            imagen,
            caption="Imagen seleccionada",
            use_container_width=True
        )


# ---------------------------------------------------------
# COLUMNA DERECHA
# ---------------------------------------------------------

with columna_resultado:

    st.subheader(
        "🔎 Resultado del análisis"
    )

    if archivo is None:

        st.info(
            "Seleccione una imagen para realizar el diagnóstico."
        )

    else:

        if st.button(
            "🚀 Analizar hoja",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Analizando la hoja..."
            ):

                enfermedad, confianza = predecir(
                    imagen
                )

                porcentaje = confianza * 100


            st.markdown(
                f"""
                <div class="resultado">

                    <div style="
                        font-size:14px;
                        color:#69766d;
                    ">
                        RESULTADO DETECTADO
                    </div>

                    <div class="enfermedad">
                        {enfermedad}
                    </div>

                    <br>

                    <div style="
                        font-size:14px;
                        color:#69766d;
                    ">
                        CONFIANZA DEL MODELO
                    </div>

                    <div class="confianza">
                        {porcentaje:.1f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            # ---------------------------------------------
            # BARRA DE CONFIANZA
            # ---------------------------------------------

            st.progress(
                min(
                    int(porcentaje),
                    100
                )
            )


            # ---------------------------------------------
            # GROQ
            # ---------------------------------------------

            st.subheader(
                "🤖 Orientación técnica"
            )

            try:

                with st.spinner(
                    "Generando recomendaciones..."
                ):

                    orientacion = generar_orientacion(
                        enfermedad,
                        porcentaje
                    )


                if orientacion:

                    st.markdown(
                        '<div class="orientacion">',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        orientacion
                    )

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True
                    )

                else:

                    st.warning(
                        "No se encontró la API Key de Groq."
                    )

            except Exception as error:

                st.error(
                    "No fue posible generar la orientación técnica."
                )

                print(
                    "Error de Groq:",
                    error
                )


# ---------------------------------------------------------
# INFORMACION DEL MODELO
# ---------------------------------------------------------

st.divider()

st.caption(
    "Modelo de clasificación basado en MobileNetV2. "
    "Clases disponibles: Minador, Phoma, Roya y Saludable."
)

st.warning(
    "El resultado corresponde a una predicción realizada "
    "por inteligencia artificial y no sustituye la evaluación "
    "de un profesional agrícola."
)
