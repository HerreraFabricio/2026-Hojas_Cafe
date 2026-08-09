import json
import os
import html
from datetime import datetime

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
# HELPER: markdown seguro para HTML indentado
# ---------------------------------------------------------
# Streamlit usa un parser de Markdown (no solo HTML puro). Ese parser:
#   1) interpreta cualquier línea indentada 4+ espacios como bloque de código, y
#   2) corta un bloque de HTML "crudo" en cuanto encuentra una línea en blanco.
# Como nuestros f-strings triple-comillas están anidados dentro de
# `with/if/else` y tienen divs anidados separados por líneas en blanco,
# textwrap.dedent() por sí solo no alcanza (solo quita la indentación común,
# no la de cada nivel de anidamiento).
#
# La solución robusta es no depender para nada de la indentación:
# aplanamos todo el HTML a una sola línea (sin saltos de línea ni
# espacios sobrantes) antes de pasarlo a st.markdown().

def render_html(contenido: str):
    lineas = [linea.strip() for linea in contenido.strip("\n").split("\n")]
    lineas_no_vacias = [linea for linea in lineas if linea]
    html_en_una_linea = " ".join(lineas_no_vacias)
    st.markdown(html_en_una_linea, unsafe_allow_html=True)


# ---------------------------------------------------------
# ESTILOS
# ---------------------------------------------------------

render_html(
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
        margin-bottom: 8px;
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
    """
)


# ---------------------------------------------------------
# CARGAR MODELO
# ---------------------------------------------------------

@st.cache_resource
def cargar_modelo():
    try:
        return tf.keras.models.load_model("mejor_modelo_cafe.keras")
    except Exception as error:
        st.error(
            "No se pudo cargar el modelo 'mejor_modelo_cafe.keras'. "
            "Verifica que el archivo exista en el repositorio."
        )
        st.exception(error)
        st.stop()


# ---------------------------------------------------------
# CARGAR CLASES
# ---------------------------------------------------------

@st.cache_data
def cargar_clases():
    try:
        with open("class_names.json", "r", encoding="utf-8") as archivo:
            return json.load(archivo)
    except Exception as error:
        st.error(
            "No se pudo cargar 'class_names.json'. "
            "Verifica que el archivo exista en el repositorio."
        )
        st.exception(error)
        st.stop()


modelo = cargar_modelo()
class_names = cargar_clases()


# ---------------------------------------------------------
# PREPARAR IMAGEN
# ---------------------------------------------------------

def preparar_imagen(imagen):
    imagen = imagen.convert("RGB")
    imagen = imagen.resize((224, 224))
    arreglo = np.array(imagen, dtype=np.float32)
    arreglo = np.expand_dims(arreglo, axis=0)
    return arreglo


# ---------------------------------------------------------
# PREDICCION
# ---------------------------------------------------------

def predecir(imagen):
    imagen_preparada = preparar_imagen(imagen)
    prediccion = modelo.predict(imagen_preparada, verbose=0)
    indice = int(np.argmax(prediccion[0]))
    confianza = float(prediccion[0][indice])
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

def generar_orientacion(enfermedad, confianza):
    api_key = obtener_groq_api_key()

    if not api_key:
        return None

    cliente = Groq(api_key=api_key)

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
    contenido = contenido.replace("```json", "").replace("```", "").strip()

    datos = json.loads(contenido)

    return datos


# ---------------------------------------------------------
# ESTADO DE LA APLICACION
# ---------------------------------------------------------

if "resultado" not in st.session_state:
    st.session_state.resultado = None

if "historial" not in st.session_state:
    st.session_state.historial = []  # lista de resultados pasados


# ---------------------------------------------------------
# COLUMNAS PRINCIPALES
# ---------------------------------------------------------

columna_imagen, columna_diagnostico = st.columns([1.02, 1], gap="large")


# ---------------------------------------------------------
# IZQUIERDA - IMAGEN
# ---------------------------------------------------------

with columna_imagen:

    render_html(
        """
        <div class="titulo-principal">
            Captura de Imagen Foliar
        </div>
        """
    )

    render_html(
        """
        <div class="subtitulo-principal">
            Posicione o cargue una hoja de café con buena iluminación.
            El sistema analizará signos visibles mediante inteligencia
            artificial.
        </div>
        """
    )

    archivo = st.file_uploader(
        "📁 Subir archivo",
        type=["jpg", "jpeg", "png"]
    )

    if archivo is not None:

        imagen = Image.open(archivo)

        st.image(imagen, use_container_width=True)

        if st.button("🚀 Analizar hoja", type="primary", use_container_width=True):

            with st.spinner("Analizando imagen..."):
                enfermedad, confianza = predecir(imagen)
                porcentaje = confianza * 100

            orientacion = None
            error_groq = None

            try:
                with st.spinner("Generando orientación técnica..."):
                    orientacion = generar_orientacion(enfermedad, porcentaje)
            except Exception as error:
                error_groq = str(error)

            resultado_actual = {
                "enfermedad": enfermedad,
                "porcentaje": porcentaje,
                "orientacion": orientacion,
                "error_groq": error_groq,
                "hora": datetime.now().strftime("%H:%M:%S"),
            }

            st.session_state.resultado = resultado_actual

            # Guardamos hasta los últimos 5 resultados como historial real
            st.session_state.historial.insert(0, resultado_actual)
            st.session_state.historial = st.session_state.historial[:5]


# ---------------------------------------------------------
# DERECHA - DIAGNOSTICO
# ---------------------------------------------------------

with columna_diagnostico:

    render_html(
        """
        <div class="mini-label">
            ÚLTIMO DIAGNÓSTICO
        </div>
        """
    )

    resultado = st.session_state.resultado

    if resultado is None:

        render_html(
            """
            <div class="resultado-nombre">
                Esperando análisis
            </div>
            """
        )

        render_html(
            """
            <div class="descripcion-modelo">
                Cargue una fotografía y presione Analizar hoja.
            </div>
            """
        )

        render_html(
            """
            <div class="aviso">
                🌿 El sistema puede clasificar hojas como
                <strong>Minador, Phoma, Roya o Saludable</strong>.
            </div>
            """
        )

    else:

        enfermedad = html.escape(resultado["enfermedad"])
        porcentaje = resultado["porcentaje"]
        orientacion = resultado["orientacion"]
        error_groq = resultado.get("error_groq")

        col_resultado, col_confianza = st.columns([3, 1])

        with col_resultado:
            render_html(
                f"""
                <div class="resultado-nombre">
                    {enfermedad}
                </div>
                """
            )

            render_html(
                """
                <div class="descripcion-modelo">
                    Resultado obtenido mediante el modelo MobileNetV2
                </div>
                """
            )

        with col_confianza:
            render_html(
                f"""
                <div class="confianza-numero">
                    {porcentaje:.1f}%
                </div>

                <div class="confianza-label">
                    CONFIANZA IA
                </div>
                """
            )

        # -------------------------------------------------
        # ORIENTACION
        # -------------------------------------------------

        if orientacion:

            descripcion = html.escape(
                orientacion.get("descripcion", "Sin información disponible.")
            )
            manejo = html.escape(
                orientacion.get("manejo", "Sin información disponible.")
            )
            buenas_practicas = html.escape(
                orientacion.get("buenas_practicas", "Sin información disponible.")
            )
            seguimiento = html.escape(
                orientacion.get("seguimiento", "Sin información disponible.")
            )

            render_html(
                f"""
                <div class="tarjeta-orientacion">

                    <div class="orientacion-header">
                        ⚠ ORIENTACIÓN Y MANEJO PREVENTIVO
                    </div>

                    <div class="orientacion-intro">
                        Orientación generada automáticamente según
                        el resultado del modelo.
                    </div>

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

                </div>
                """
            )

        else:
            mensaje = "No fue posible generar las recomendaciones con Groq."
            if error_groq:
                mensaje += f" Detalle técnico: {error_groq}"
            st.warning(mensaje)

        # -------------------------------------------------
        # HISTORIAL (real, no solo el último resultado)
        # -------------------------------------------------

        render_html(
            """
            <div class="historial-titulo">
                HISTORIAL RECIENTE
            </div>
            """
        )

        for item in st.session_state.historial:
            nombre_item = html.escape(item["enfermedad"])
            render_html(
                f"""
                <div class="historial-box">
                    <span class="historial-punto">●</span>
                    <strong>{nombre_item}</strong>
                    <span style="float:right; color:#8a8178;">
                        {item['porcentaje']:.1f}% · {item['hora']}
                    </span>
                </div>
                """
            )

        render_html(
            """
            <div class="aviso">
                ⚠ La clasificación es una predicción realizada
                mediante inteligencia artificial y no sustituye
                la evaluación de un técnico agrícola.
            </div>
            """
        )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

render_html(
    """
    <div class="footer-personalizado">
        © 2026 · DETECCIÓN DE ENFERMEDADES EN CAFÉ ·
        MOBILE NETV2 + GROQ API
    </div>
    """
)
