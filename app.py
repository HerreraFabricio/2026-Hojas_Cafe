import json
import os
import html

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

    /* Fondo general */
    .stApp {
        background-color: #fbf8f3;
    }

    /* Contenedor principal */
    .block-container {
        max-width: 1500px;
        padding-top: 1.3rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Ocultar algunos elementos de Streamlit */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* Titulos */
    .titulo-principal {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 39px;
        font-weight: 700;
        color: #21160f;
        margin-bottom: 8px;
        line-height: 1.1;
    }

    .subtitulo-principal {
        font-size: 14px;
        color: #746b62;
        margin-bottom: 25px;
        line-height: 1.6;
    }

    .mini-label {
        font-size: 10px;
        letter-spacing: 1.6px;
        color: #81766b;
        font-weight: 800;
        margin-bottom: 12px;
        text-transform: uppercase;
    }

    /* Resultado */
    .resultado-nombre {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 31px;
        font-weight: 700;
        color: #2c190d;
        line-height: 1.2;
        margin-top: 3px;
        margin-bottom: 6px;
    }

    .descripcion-modelo {
        color: #85786e;
        font-size: 11px;
        font-style: italic;
    }

    .confianza-numero {
        font-family: Georgia, "Times New Roman", serif;
        text-align: right;
        font-size: 35px;
        font-weight: 700;
        color: #29170d;
        line-height: 1;
        margin-top: 3px;
    }

    .confianza-label {
        text-align: right;
        font-size: 9px;
        letter-spacing: 1px;
        font-weight: 800;
        color: #4f443b;
        margin-top: 6px;
    }

    /* Tarjeta orientación */
    .tarjeta-orientacion {
        background-color: #f1ece4;
        border: 1px solid #e0d7ca;
        border-radius: 18px;
        padding: 20px 24px;
        margin-top: 20px;
        margin-bottom: 25px;
    }

    .orientacion-header {
        font-size: 10px;
        letter-spacing: 1.4px;
        color: #746a60;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .orientacion-intro {
        font-size: 12px;
        color: #50483f;
        margin-bottom: 8px;
    }

    /* Elementos numerados */
    .recomendacion-item {
        padding: 15px 0;
        border-bottom: 1px solid #ded5c9;
    }

    .recomendacion-item:last-child {
        border-bottom: none;
    }

    .numero {
        display: inline-block;
        min-width: 34px;
        padding: 7px 7px;
        margin-right: 12px;
        background-color: #2f6545;
        color: #ffffff;
        border-radius: 8px;
        text-align: center;
        font-size: 11px;
        font-weight: 800;
        vertical-align: top;
    }

    .recomendacion-contenido {
        display: inline-block;
        width: calc(100% - 55px);
        vertical-align: top;
    }

    .recomendacion-titulo {
        font-size: 13px;
        font-weight: 800;
        color: #2a2019;
        margin-bottom: 6px;
    }

    .recomendacion-texto {
        font-size: 12px;
        line-height: 1.65;
        color: #4e453d;
    }

    /* Historial */
    .historial-titulo {
        margin-top: 24px;
        margin-bottom: 10px;
        font-size: 10px;
        letter-spacing: 1.6px;
        color: #81766b;
        font-weight: 800;
    }

    .historial-box {
        background-color: #ffffff;
        border: 1px solid #ded6cc;
        border-radius: 11px;
        padding: 13px 16px;
        font-size: 12px;
        color: #38291f;
    }

    .historial-punto {
        color: #c66b1d;
        margin-right: 8px;
    }

    /* Disclaimer */
    .aviso {
        background-color: #fffaf0;
        border: 1px solid #e7dfce;
        border-radius: 12px;
        padding: 13px 15px;
        color: #675b50;
        font-size: 11px;
        margin-top: 20px;
    }

    /* Footer */
    .footer-personalizado {
        margin-top: 26px;
        border-top: 1px solid #ded6ca;
        padding-top: 20px;
        color: #92887e;
        font-size: 10px;
    }

    /* Botones */
    div.stButton > button {
        background-color: #321d10;
        color: white;
        border: none;
        border-radius: 22px;
        width: 100%;
        min-height: 45px;
        font-weight: 700;
    }

    div.stButton > button:hover {
        background-color: #4b2b18;
        color: white;
        border: none;
    }

    /* Uploader */
    [data-testid="stFileUploader"] {
        background-color: transparent;
    }

    /* Imagen */
    [data-testid="stImage"] img {
        border-radius: 6px;
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
# CARGAR CLASES
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
# PREDICCION
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
# OBTENER API KEY GROQ
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
Actúa como asistente técnico especializado en cultivo de café.

Un modelo de inteligencia artificial analizó una fotografía
de una hoja de café.

Resultado detectado: {enfermedad}
Confianza del modelo: {confianza:.1f}%

Devuelve ÚNICAMENTE un JSON válido con exactamente esta estructura:

{{
    "descripcion": "Texto breve explicando el resultado.",
    "manejo": "Recomendaciones preventivas y de manejo.",
    "buenas_practicas": "Buenas prácticas para el cuidado del cultivo.",
    "seguimiento": "Acciones de seguimiento y monitoreo."
}}

Reglas:

- Escribe en español.
- Usa lenguaje claro para un productor de café.
- No afirmes que la predicción sustituye un diagnóstico profesional.
- No inventes productos químicos ni dosis.
- Si el resultado es Saludable, da recomendaciones para mantener
  la planta en buenas condiciones.
- Si los síntomas parecen graves o continúan avanzando,
  recomienda consultar a un técnico agrícola.
- No escribas nada antes ni después del JSON.
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
        temperature=0.3,
        max_tokens=900
    )

    contenido = respuesta.choices[0].message.content.strip()

    # Por si Groq coloca ```json
    contenido = contenido.replace(
        "```json",
        ""
    ).replace(
        "```",
        ""
    ).strip()

    datos = json.loads(
        contenido
    )

    return datos


# ---------------------------------------------------------
# ESTADO DE LA APLICACION
# ---------------------------------------------------------

if "resultado" not in st.session_state:
    st.session_state.resultado = None


# ---------------------------------------------------------
# COLUMNAS PRINCIPALES
# ---------------------------------------------------------

columna_imagen, columna_diagnostico = st.columns(
    [1.02, 1],
    gap="large"
)


# ---------------------------------------------------------
# IZQUIERDA - IMAGEN
# ---------------------------------------------------------

with columna_imagen:

    st.markdown(
        '<div class="titulo-principal">'
        'Captura de Imagen Foliar'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitulo-principal">
            Posicione o cargue una hoja de café con buena iluminación.
            El sistema analizará signos visibles mediante inteligencia
            artificial.
        </div>
        """,
        unsafe_allow_html=True
    )

    archivo = st.file_uploader(
        "📁 Subir archivo",
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
            use_container_width=True
        )

        if st.button(
            "🚀 Analizar hoja",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Analizando imagen..."
            ):

                enfermedad, confianza = predecir(
                    imagen
                )

                porcentaje = confianza * 100

            try:

                with st.spinner(
                    "Generando orientación técnica..."
                ):

                    orientacion = generar_orientacion(
                        enfermedad,
                        porcentaje
                    )

            except Exception as error:

                orientacion = None

                print(
                    "Error de Groq:",
                    error
                )

            st.session_state.resultado = {
                "enfermedad": enfermedad,
                "porcentaje": porcentaje,
                "orientacion": orientacion
            }


# ---------------------------------------------------------
# DERECHA - DIAGNOSTICO
# ---------------------------------------------------------

with columna_diagnostico:

    st.markdown(
        '<div class="mini-label">'
        'ÚLTIMO DIAGNÓSTICO'
        '</div>',
        unsafe_allow_html=True
    )

    resultado = st.session_state.resultado

    if resultado is None:

        st.markdown(
            '<div class="resultado-nombre">'
            'Esperando análisis'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="descripcion-modelo">'
            'Cargue una fotografía y presione Analizar hoja.'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="aviso">
                🌿 El sistema puede clasificar hojas como
                <strong>Minador, Phoma, Roya o Saludable</strong>.
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        enfermedad = html.escape(
            resultado["enfermedad"]
        )

        porcentaje = resultado["porcentaje"]

        orientacion = resultado["orientacion"]

        col_resultado, col_confianza = st.columns(
            [3, 1]
        )

        with col_resultado:

            st.markdown(
                f"""
                <div class="resultado-nombre">
                    {enfermedad}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="descripcion-modelo">
                    Resultado obtenido mediante el modelo MobileNetV2
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_confianza:

            st.markdown(
                f"""
                <div class="confianza-numero">
                    {porcentaje:.1f}%
                </div>

                <div class="confianza-label">
                    CONFIANZA IA
                </div>
                """,
                unsafe_allow_html=True
            )

        # -------------------------------------------------
        # ORIENTACION
        # -------------------------------------------------

        st.markdown(
            """
            <div class="tarjeta-orientacion">

                <div class="orientacion-header">
                    ⚠ ORIENTACIÓN Y MANEJO PREVENTIVO
                </div>

                <div class="orientacion-intro">
                    Orientación generada automáticamente según
                    el resultado del modelo.
                </div>
            """,
            unsafe_allow_html=True
        )

        if orientacion:

            descripcion = html.escape(
                orientacion.get(
                    "descripcion",
                    "Sin información disponible."
                )
            )

            manejo = html.escape(
                orientacion.get(
                    "manejo",
                    "Sin información disponible."
                )
            )

            buenas_practicas = html.escape(
                orientacion.get(
                    "buenas_practicas",
                    "Sin información disponible."
                )
            )

            seguimiento = html.escape(
                orientacion.get(
                    "seguimiento",
                    "Sin información disponible."
                )
            )

            st.markdown(
                f"""
                <div class="recomendacion-item">

                    <span class="numero">01</span>

                    <div class="recomendacion-contenido">

                        <div class="recomendacion-titulo">
                            Descripción del resultado
                        </div>

                        <div class="recomendacion-texto">
                            {descripcion}
                        </div>

                    </div>

                </div>


                <div class="recomendacion-item">

                    <span class="numero">02</span>

                    <div class="recomendacion-contenido">

                        <div class="recomendacion-titulo">
                            Manejo preventivo
                        </div>

                        <div class="recomendacion-texto">
                            {manejo}
                        </div>

                    </div>

                </div>


                <div class="recomendacion-item">

                    <span class="numero">03</span>

                    <div class="recomendacion-contenido">

                        <div class="recomendacion-titulo">
                            Buenas prácticas
                        </div>

                        <div class="recomendacion-texto">
                            {buenas_practicas}
                        </div>

                    </div>

                </div>


                <div class="recomendacion-item">

                    <span class="numero">04</span>

                    <div class="recomendacion-contenido">

                        <div class="recomendacion-titulo">
                            Monitoreo y seguimiento
                        </div>

                        <div class="recomendacion-texto">
                            {seguimiento}
                        </div>

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.warning(
                "No fue posible generar las recomendaciones "
                "con Groq."
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

        # -------------------------------------------------
        # HISTORIAL
        # -------------------------------------------------

        st.markdown(
            '<div class="historial-titulo">'
            'HISTORIAL RECIENTE'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="historial-box">

                <span class="historial-punto">
                    ●
                </span>

                <strong>
                    {enfermedad}
                </strong>

                <span style="
                    float:right;
                    color:#8a8178;
                ">
                    {porcentaje:.1f}% confianza
                </span>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="aviso">
                ⚠ La clasificación es una predicción realizada
                mediante inteligencia artificial y no sustituye
                la evaluación de un técnico agrícola.
            </div>
            """,
            unsafe_allow_html=True
        )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="footer-personalizado">
        © 2026 · DETECCIÓN DE ENFERMEDADES EN CAFÉ ·
        MOBILE NETV2 + GROQ API
    </div>
    """,
    unsafe_allow_html=True
)
