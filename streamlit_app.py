import re
import html
import zipfile
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import streamlit as st
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Analítica de Sentimientos Yelp",
    page_icon="💬",
    layout="wide"
)

MODEL_ZIP_URL = (
    "https://github.com/"
    "mtdigital2017-dotcom/"
    "Caso-Practico-Yelp-Sentimiento/"
    "releases/download/v1.0/"
    "yelp_bert_deployment.zip"
)

CACHE_DIR = Path("/tmp/yelp_model")
MODEL_DIR = CACHE_DIR / "bert_model"
TOKENIZER_DIR = CACHE_DIR / "tokenizer"

MAX_LENGTH = 256

ID2LABEL = {
    0: "NEGATIVO",
    1: "NEUTRAL",
    2: "POSITIVO"
}


# ============================================================
# PREPARAR ARCHIVOS DEL MODELO
# ============================================================

def prepare_model_files():

    model_ready = (
        (MODEL_DIR / "config.json").exists()
        and (
            (MODEL_DIR / "model.safetensors").exists()
            or
            (MODEL_DIR / "pytorch_model.bin").exists()
        )
    )

    tokenizer_ready = (
        (TOKENIZER_DIR / "tokenizer_config.json").exists()
    )

    if model_ready and tokenizer_ready:
        return

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    zip_path = (
        CACHE_DIR
        / "yelp_bert_deployment.zip"
    )

    with st.spinner(
        "Descargando modelo BERT. "
        "La primera carga puede tardar unos minutos..."
    ):

        response = requests.get(
            MODEL_ZIP_URL,
            stream=True,
            timeout=600
        )

        response.raise_for_status()

        with open(zip_path, "wb") as f:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:
                    f.write(chunk)

        with zipfile.ZipFile(
            zip_path,
            "r"
        ) as z:

            z.extractall(
                CACHE_DIR
            )

        zip_path.unlink(
            missing_ok=True
        )


# ============================================================
# CARGAR MODELO
# ============================================================

@st.cache_resource
def load_model():

    prepare_model_files()

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            TOKENIZER_DIR
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_DIR
        )
    )

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()


# ============================================================
# LIMPIEZA DE TEXTO
# ============================================================

def clean_text(text):

    text = html.unescape(
        str(text)
    )

    text = re.sub(
        r"[\x00-\x1F\x7F]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# INFERENCIA
# ============================================================

def predict_batch(
    texts,
    batch_size=4
):

    model.eval()

    predictions = []
    probabilities = []

    for start in range(
        0,
        len(texts),
        batch_size
    ):

        batch_texts = [
            clean_text(x)
            for x in texts[
                start:
                start + batch_size
            ]
        ]

        encoded = tokenizer(
            batch_texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt"
        )

        with torch.no_grad():

            logits = model(
                **encoded
            ).logits

            probs = torch.softmax(
                logits,
                dim=1
            )

            preds = torch.argmax(
                probs,
                dim=1
            )

        predictions.extend(
            preds.cpu()
            .numpy()
            .tolist()
        )

        probabilities.extend(
            probs.cpu()
            .numpy()
            .tolist()
        )

    return (
        np.asarray(predictions),
        np.asarray(probabilities)
    )


# ============================================================
# TEMAS NEGATIVOS
# ============================================================

CATEGORIAS = {

    "Comida / producto": [
        "food",
        "meal",
        "dish",
        "taste",
        "cold",
        "dry",
        "bad",
        "bland",
        "overcooked",
        "undercooked"
    ],

    "Servicio / atención": [
        "service",
        "staff",
        "waiter",
        "waitress",
        "server",
        "manager",
        "employee",
        "rude",
        "customer service"
    ],

    "Tiempo / demora": [
        "wait",
        "waiting",
        "slow",
        "minutes",
        "hour",
        "late",
        "forever"
    ],

    "Precio / valor": [
        "price",
        "expensive",
        "overpriced",
        "cost",
        "money",
        "worth"
    ],

    "Limpieza / instalaciones": [
        "dirty",
        "clean",
        "bathroom",
        "restroom",
        "table",
        "smell",
        "parking"
    ],

    "Pedido / entrega": [
        "order",
        "delivery",
        "delivered",
        "wrong order",
        "takeout",
        "take out"
    ]
}


def detect_theme(
    text,
    words
):

    text = str(text).lower()

    return any(
        re.search(
            r"\b"
            + re.escape(word)
            + r"\b",
            text
        )
        for word in words
    )


# ============================================================
# ESTRATEGIAS DE COMUNICACIÓN
# ============================================================

STRATEGIES = {

    "Comida / producto": {
        "estrategia":
            "Comunicación de mejora del producto y recuperación de confianza",
        "accion":
            "Informar acciones de control de calidad y responder casos críticos",
        "canal":
            "Respuestas a reseñas, redes sociales y comunicación en punto de venta",
        "mensaje":
            "Estamos trabajando para mejorar de manera continua la calidad de nuestros productos y tu experiencia.",
        "indicador":
            "% de reseñas negativas asociadas a producto"
    },

    "Servicio / atención": {
        "estrategia":
            "Recuperación del servicio y fortalecimiento de la atención",
        "accion":
            "Responder comentarios negativos y comunicar acciones de mejora en atención",
        "canal":
            "Yelp, redes sociales, atención directa y correo posvisita",
        "mensaje":
            "Tus comentarios nos ayudan a mejorar. Estamos fortaleciendo nuestros procesos de atención para ofrecerte una mejor experiencia.",
        "indicador":
            "% de reseñas negativas asociadas a servicio"
    },

    "Tiempo / demora": {
        "estrategia":
            "Gestión de expectativas y comunicación de tiempos",
        "accion":
            "Informar tiempos estimados y comunicar medidas para reducir esperas",
        "canal":
            "Punto de atención, web, redes y mensajes transaccionales",
        "mensaje":
            "Estamos trabajando para reducir los tiempos de espera y ofrecerte una atención más ágil.",
        "indicador":
            "% de reseñas negativas asociadas a demora"
    },

    "Precio / valor": {
        "estrategia":
            "Refuerzo de propuesta de valor",
        "accion":
            "Comunicar beneficios, atributos diferenciales y relación calidad-precio",
        "canal":
            "Redes sociales, sitio web, promociones y comunicaciones comerciales",
        "mensaje":
            "Queremos que cada experiencia refleje el valor, la calidad y el servicio que esperas.",
        "indicador":
            "% de reseñas negativas asociadas a precio/valor"
    },

    "Limpieza / instalaciones": {
        "estrategia":
            "Comunicación de confianza y estándares operativos",
        "accion":
            "Reforzar protocolos y comunicar mejoras visibles en instalaciones",
        "canal":
            "Punto de atención, redes y respuestas a reseñas",
        "mensaje":
            "La limpieza y el cuidado de nuestros espacios son parte esencial de la experiencia que queremos ofrecer.",
        "indicador":
            "% de reseñas negativas asociadas a instalaciones"
    },

    "Pedido / entrega": {
        "estrategia":
            "Comunicación de confiabilidad del proceso de pedido",
        "accion":
            "Informar mejoras en preparación, validación y entrega de pedidos",
        "canal":
            "Mensajes transaccionales, soporte y respuestas directas",
        "mensaje":
            "Estamos fortaleciendo nuestros procesos para que tus pedidos lleguen correctamente y a tiempo.",
        "indicador":
            "% de reseñas negativas asociadas a pedido/entrega"
    }
}


def communication_priority(
    negative_pct
):

    if negative_pct >= 30:
        return "🔴 ALTA"

    elif negative_pct >= 15:
        return "🟠 MEDIA"

    else:
        return "🟢 BAJA"


def global_diagnosis(
    positive_pct,
    neutral_pct,
    negative_pct
):

    if negative_pct >= 30:

        return (
            "La proporción de sentimiento negativo es elevada. "
            "Se recomienda priorizar acciones de recuperación, "
            "escucha activa y comunicación correctiva."
        )

    elif negative_pct >= 15:

        return (
            "La percepción general es favorable, pero existe un "
            "segmento relevante de experiencias negativas que "
            "requiere seguimiento y comunicación focalizada."
        )

    else:

        return (
            "La percepción general es favorable. "
            "La estrategia puede centrarse en fidelización, "
            "amplificación de experiencias positivas y seguimiento "
            "preventivo de los temas negativos."
        )


# ============================================================
# ENCABEZADO
# ============================================================

st.title(
    "💬 Analítica de sentimientos en reseñas de Yelp"
)

st.caption(
    "Aplicación BERT de despliegue"
)

st.markdown(
    """
Clasificación de reseñas en:

**NEGATIVO · NEUTRAL · POSITIVO**

La aplicación utiliza exclusivamente
el texto como predictor.
"""
)


# ============================================================
# PESTAÑAS
# ============================================================

tab_individual, tab_masivo = st.tabs(
    [
        "Análisis individual",
        "Análisis masivo"
    ]
)


# ============================================================
# ANÁLISIS INDIVIDUAL
# ============================================================

with tab_individual:

    review = st.text_area(
        "Escribe una reseña en inglés:",
        height=150
    )

    if st.button(
        "Analizar sentimiento",
        type="primary"
    ):

        if not review.strip():

            st.warning(
                "Introduce una reseña."
            )

        else:

            pred, probs = predict_batch(
                [review],
                batch_size=1
            )

            pred_id = int(
                pred[0]
            )

            p = probs[0]

            st.subheader(
                f"Resultado: {ID2LABEL[pred_id]}"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "NEGATIVO",
                f"{p[0] * 100:.2f}%"
            )

            col2.metric(
                "NEUTRAL",
                f"{p[1] * 100:.2f}%"
            )

            col3.metric(
                "POSITIVO",
                f"{p[2] * 100:.2f}%"
            )

            st.write(
                "Confianza del modelo:",
                f"{np.max(p) * 100:.2f}%"
            )


# ============================================================
# ANÁLISIS MASIVO
# ============================================================

with tab_masivo:

    uploaded_file = st.file_uploader(
        "Sube un archivo CSV o Excel",
        type=[
            "csv",
            "xlsx"
        ]
    )

    if uploaded_file is not None:

        if uploaded_file.name.lower().endswith(
            ".csv"
        ):

            df = pd.read_csv(
                uploaded_file
            )

        else:

            df = pd.read_excel(
                uploaded_file
            )

        st.write(
            f"Filas cargadas: {len(df):,}"
        )

        if "text" not in df.columns:

            st.error(
                "El archivo debe contener "
                "una columna llamada 'text'."
            )

        else:

            st.info(
                "La versión pública procesa una muestra "
                "controlada para evitar exceder los "
                "recursos gratuitos de Streamlit."
            )

            opciones = [
                100,
                250,
                500,
                1000
            ]

            max_disponible = len(df)

            opciones_validas = [
                x
                for x in opciones
                if x <= max_disponible
            ]

            if not opciones_validas:
                opciones_validas = [
                    max_disponible
                ]

            n_analisis = st.selectbox(
                "Número de reseñas a analizar:",
                options=opciones_validas,
                index=0
            )

            metodo = st.radio(
                "Selección de reseñas:",
                [
                    "Primeras filas",
                    "Muestra aleatoria reproducible"
                ]
            )

            if st.button(
                "Analizar dataset",
                type="primary"
            ):

                work = df[
                    df["text"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .ne("")
                ].copy()

                if (
                    metodo
                    ==
                    "Muestra aleatoria reproducible"
                ):

                    work = work.sample(
                        n=min(
                            n_analisis,
                            len(work)
                        ),
                        random_state=42
                    )

                else:

                    work = work.head(
                        n_analisis
                    )

                work = work.reset_index(
                    drop=True
                )

                st.write(
                    f"Reseñas que serán analizadas: "
                    f"{len(work):,}"
                )

                progress = st.progress(
                    0
                )

                status = st.empty()

                all_predictions = []
                all_probabilities = []

                batch_size = 4

                texts = (
                    work["text"]
                    .astype(str)
                    .tolist()
                )

                total = len(
                    texts
                )

                for start in range(
                    0,
                    total,
                    batch_size
                ):

                    batch = texts[
                        start:
                        start + batch_size
                    ]

                    pred, probs = predict_batch(
                        batch,
                        batch_size=batch_size
                    )

                    all_predictions.extend(
                        pred.tolist()
                    )

                    all_probabilities.extend(
                        probs.tolist()
                    )

                    processed = min(
                        start + batch_size,
                        total
                    )

                    progress.progress(
                        processed / total
                    )

                    status.write(
                        f"Procesando "
                        f"{processed:,} / "
                        f"{total:,}"
                    )

                pred = np.asarray(
                    all_predictions
                )

                probs = np.asarray(
                    all_probabilities
                )

                work[
                    "sentiment_predicted"
                ] = [
                    ID2LABEL[
                        int(x)
                    ]
                    for x in pred
                ]

                work[
                    "confidence"
                ] = probs.max(
                    axis=1
                )

                work[
                    "prob_NEGATIVO"
                ] = probs[:, 0]

                work[
                    "prob_NEUTRAL"
                ] = probs[:, 1]

                work[
                    "prob_POSITIVO"
                ] = probs[:, 2]

                counts = (
                    work[
                        "sentiment_predicted"
                    ]
                    .value_counts()
                    .reindex(
                        [
                            "NEGATIVO",
                            "NEUTRAL",
                            "POSITIVO"
                        ],
                        fill_value=0
                    )
                )

                percentages = (
                    counts
                    / len(work)
                    * 100
                )

                st.success(
                    "Análisis completado."
                )

                # =================================================
                # DISTRIBUCIÓN
                # =================================================

                st.header(
                    "Distribución de sentimientos"
                )

                col1, col2, col3 = st.columns(
                    3
                )

                col1.metric(
                    "NEGATIVO",
                    f"{percentages['NEGATIVO']:.2f}%",
                    f"{counts['NEGATIVO']:,} reseñas"
                )

                col2.metric(
                    "NEUTRAL",
                    f"{percentages['NEUTRAL']:.2f}%",
                    f"{counts['NEUTRAL']:,} reseñas"
                )

                col3.metric(
                    "POSITIVO",
                    f"{percentages['POSITIVO']:.2f}%",
                    f"{counts['POSITIVO']:,} reseñas"
                )

                chart_df = pd.DataFrame({
                    "Porcentaje":
                        percentages
                })

                st.bar_chart(
                    chart_df
                )

                # =================================================
                # TEMAS NEGATIVOS
                # =================================================

                negativas = work[
                    work[
                        "sentiment_predicted"
                    ]
                    ==
                    "NEGATIVO"
                ].copy()

                causas = []

                for categoria, palabras in CATEGORIAS.items():

                    cantidad = int(
                        negativas[
                            "text"
                        ]
                        .apply(
                            lambda x:
                            detect_theme(
                                x,
                                palabras
                            )
                        )
                        .sum()
                    )

                    porcentaje = (
                        cantidad
                        / len(negativas)
                        * 100
                        if len(negativas)
                        else 0
                    )

                    causas.append({
                        "Tema":
                            categoria,
                        "Reseñas":
                            cantidad,
                        "% de negativas":
                            round(
                                porcentaje,
                                2
                            )
                    })

                resumen_causas = (
                    pd.DataFrame(
                        causas
                    )
                    .sort_values(
                        "Reseñas",
                        ascending=False
                    )
                    .reset_index(
                        drop=True
                    )
                )

                st.header(
                    "Principales temas en reseñas negativas"
                )

                st.dataframe(
                    resumen_causas,
                    use_container_width=True
                )

                # =================================================
                # ANÁLISIS EJECUTIVO
                # =================================================

                confianza_promedio = (
                    work[
                        "confidence"
                    ].mean()
                    * 100
                )

                baja_confianza = (
                    (
                        work[
                            "confidence"
                        ] < 0.60
                    )
                    .mean()
                    * 100
                )

                dominante = (
                    counts.idxmax()
                )

                st.header(
                    "Análisis ejecutivo"
                )

                st.markdown(
                    f"""
Se analizaron **{len(work):,} reseñas**.

El sentimiento predominante fue
**{dominante}**.

La distribución obtenida fue:

- **POSITIVO:** {percentages['POSITIVO']:.2f} %
- **NEUTRAL:** {percentages['NEUTRAL']:.2f} %
- **NEGATIVO:** {percentages['NEGATIVO']:.2f} %

La confianza promedio del modelo fue
**{confianza_promedio:.2f} %**.

El **{baja_confianza:.2f} %**
de las predicciones tuvo confianza
inferior al 60 % y debería considerarse
para revisión manual.
"""
                )

                # =================================================
                # DIAGNÓSTICO DE COMUNICACIÓN
                # =================================================

                st.header(
                    "📣 Diagnóstico y estrategia de comunicación"
                )

                negative_pct = float(
                    percentages[
                        "NEGATIVO"
                    ]
                )

                positive_pct = float(
                    percentages[
                        "POSITIVO"
                    ]
                )

                neutral_pct = float(
                    percentages[
                        "NEUTRAL"
                    ]
                )

                priority = (
                    communication_priority(
                        negative_pct
                    )
                )

                diagnosis = (
                    global_diagnosis(
                        positive_pct,
                        neutral_pct,
                        negative_pct
                    )
                )

                st.subheader(
                    f"Prioridad general: {priority}"
                )

                st.write(
                    diagnosis
                )

                # =================================================
                # PLAN DE COMUNICACIÓN POR TEMA
                # =================================================

                communication_rows = []

                for _, row in resumen_causas.iterrows():

                    theme = row[
                        "Tema"
                    ]

                    theme_pct = float(
                        row[
                            "% de negativas"
                        ]
                    )

                    theme_count = int(
                        row[
                            "Reseñas"
                        ]
                    )

                    strategy = STRATEGIES[
                        theme
                    ]

                    if theme_count == 0:

                        theme_priority = (
                            "⚪ SIN SEÑAL"
                        )

                    elif theme_pct >= 30:

                        theme_priority = (
                            "🔴 ALTA"
                        )

                    elif theme_pct >= 15:

                        theme_priority = (
                            "🟠 MEDIA"
                        )

                    else:

                        theme_priority = (
                            "🟢 BAJA"
                        )

                    communication_rows.append({

                        "Prioridad":
                            theme_priority,

                        "Tema":
                            theme,

                        "% negativas":
                            round(
                                theme_pct,
                                2
                            ),

                        "Estrategia":
                            strategy[
                                "estrategia"
                            ],

                        "Acción recomendada":
                            strategy[
                                "accion"
                            ],

                        "Canal sugerido":
                            strategy[
                                "canal"
                            ],

                        "Mensaje sugerido":
                            strategy[
                                "mensaje"
                            ],

                        "Indicador":
                            strategy[
                                "indicador"
                            ]
                    })

                communication_plan = (
                    pd.DataFrame(
                        communication_rows
                    )
                )

                st.subheader(
                    "Plan de comunicación recomendado"
                )

                st.dataframe(
                    communication_plan,
                    use_container_width=True,
                    hide_index=True
                )

                # =================================================
                # RECOMENDACIONES GENERALES
                # =================================================

                st.subheader(
                    "Recomendaciones generales"
                )

                if negative_pct >= 30:

                    st.markdown(
                        """
**Objetivo principal:** recuperación de confianza.

- Priorizar respuesta a comentarios negativos.
- Comunicar acciones correctivas concretas.
- Dar seguimiento semanal a los temas críticos.
- Revisar manualmente casos de baja confianza.
"""
                    )

                elif negative_pct >= 15:

                    st.markdown(
                        """
**Objetivo principal:** fortalecer la experiencia y reducir fricciones.

- Mantener comunicación positiva con clientes satisfechos.
- Atender de forma focalizada los principales temas negativos.
- Comunicar mejoras implementadas.
- Dar seguimiento a la evolución del sentimiento negativo.
"""
                    )

                else:

                    st.markdown(
                        """
**Objetivo principal:** fidelización y amplificación.

- Amplificar experiencias positivas.
- Promover testimonios y recomendaciones.
- Mantener seguimiento preventivo de señales negativas.
- Reforzar atributos bien valorados.
"""
                    )

                # =================================================
                # NOTA METODOLÓGICA
                # =================================================

                st.info(
                    """
Las estrategias de comunicación son recomendaciones
derivadas de reglas analíticas aplicadas sobre los
resultados agregados.

BERT clasifica el sentimiento. Las estrategias no son
predicciones del modelo y deben interpretarse como apoyo
a la toma de decisiones.

Los temas negativos se identifican mediante palabras clave;
su presencia indica asociación temática y no demuestra
causalidad.
"""
                )

                # =================================================
                # DESCARGAS
                # =================================================

                st.header(
                    "Descargas"
                )

                col_download_1, col_download_2 = st.columns(
                    2
                )

                predictions_csv = (
                    work.to_csv(
                        index=False
                    )
                    .encode(
                        "utf-8"
                    )
                )

                communication_csv = (
                    communication_plan
                    .to_csv(
                        index=False
                    )
                    .encode(
                        "utf-8"
                    )
                )

                with col_download_1:

                    st.download_button(
                        "Descargar predicciones",
                        data=predictions_csv,
                        file_name=(
                            "analisis_sentimientos_yelp.csv"
                        ),
                        mime="text/csv"
                    )

                with col_download_2:

                    st.download_button(
                        "Descargar plan de comunicación",
                        data=communication_csv,
                        file_name=(
                            "plan_comunicacion_yelp.csv"
                        ),
                        mime="text/csv"
                    )

                st.caption(
                    "La versión pública utiliza una muestra "
                    "controlada por limitaciones de recursos "
                    "del servicio gratuito de despliegue."
                )
