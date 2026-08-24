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

        with open(
            zip_path,
            "wb"
        ) as f:

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

                if metodo == "Muestra aleatoria reproducible":

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

                # ------------------------------------------------
                # DISTRIBUCIÓN
                # ------------------------------------------------

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

                # ------------------------------------------------
                # NEGATIVAS
                # ------------------------------------------------

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

                # ------------------------------------------------
                # ANÁLISIS EJECUTIVO
                # ------------------------------------------------

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

El **{baja_confianza:.2f} %** de las predicciones
tuvo confianza inferior al 60 % y debería
considerarse para revisión manual.

Los temas negativos se identifican mediante
reglas de palabras clave. Su presencia indica
asociación temática y no demuestra causalidad.

> Esta aplicación pública utiliza una muestra
> controlada por limitaciones de recursos del
> servicio gratuito de despliegue.
"""
                )

                # ------------------------------------------------
                # DESCARGA
                # ------------------------------------------------

                csv_data = (
                    work.to_csv(
                        index=False
                    )
                    .encode(
                        "utf-8"
                    )
                )

                st.download_button(
                    "Descargar predicciones",
                    data=csv_data,
                    file_name=(
                        "analisis_sentimientos_yelp.csv"
                    ),
                    mime="text/csv"
                )
