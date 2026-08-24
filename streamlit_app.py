
import re
import html
import zipfile
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import streamlit as st
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification


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


def prepare_model_files():

    if (
        (MODEL_DIR / "config.json").exists()
        and
        (TOKENIZER_DIR / "tokenizer_config.json").exists()
    ):
        return

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    zip_path = CACHE_DIR / "yelp_bert_deployment.zip"

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

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(CACHE_DIR)

    zip_path.unlink(
        missing_ok=True
    )


@st.cache_resource
def load_model():

    prepare_model_files()

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_DIR
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()


def clean_text(text):

    text = html.unescape(str(text))

    text = re.sub(
        r"[\x00-\x1F\x7F]",
        " ",
        text
    )

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def predict_batch(texts, batch_size=8):

    preds_all = []
    probs_all = []

    for start in range(0, len(texts), batch_size):

        batch = [
            clean_text(x)
            for x in texts[start:start + batch_size]
        ]

        enc = tokenizer(
            batch,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt"
        )

        with torch.no_grad():

            logits = model(**enc).logits

            probs = torch.softmax(
                logits,
                dim=1
            )

            preds = torch.argmax(
                probs,
                dim=1
            )

        preds_all.extend(
            preds.numpy().tolist()
        )

        probs_all.extend(
            probs.numpy().tolist()
        )

    return (
        np.array(preds_all),
        np.array(probs_all)
    )


CATEGORIAS = {

    "Comida / producto": [
        "food", "meal", "dish", "taste",
        "cold", "dry", "bad", "bland",
        "overcooked", "undercooked"
    ],

    "Servicio / atención": [
        "service", "staff", "waiter",
        "waitress", "server", "manager",
        "employee", "rude", "customer service"
    ],

    "Tiempo / demora": [
        "wait", "waiting", "slow",
        "minutes", "hour", "late", "forever"
    ],

    "Precio / valor": [
        "price", "expensive", "overpriced",
        "cost", "money", "worth"
    ],

    "Limpieza / instalaciones": [
        "dirty", "clean", "bathroom",
        "restroom", "table", "smell", "parking"
    ],

    "Pedido / entrega": [
        "order", "delivery", "delivered",
        "wrong order", "takeout", "take out"
    ]
}


def detect_theme(text, words):

    text = str(text).lower()

    return any(
        re.search(
            r"\b" + re.escape(w) + r"\b",
            text
        )
        for w in words
    )


st.title(
    "💬 Analítica de sentimientos en reseñas de Yelp"
)

st.write(
    "Clasificación BERT en NEGATIVO, NEUTRAL y POSITIVO."
)


tab1, tab2 = st.tabs(
    ["Análisis individual", "Análisis masivo"]
)


with tab1:

    text = st.text_area(
        "Escribe una reseña en inglés:"
    )

    if st.button(
        "Analizar sentimiento"
    ):

        if text.strip():

            pred, probs = predict_batch([text])

            p = probs[0]

            st.subheader(
                f"Resultado: {ID2LABEL[int(pred[0])]}"
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "NEGATIVO",
                f"{p[0]*100:.2f}%"
            )

            c2.metric(
                "NEUTRAL",
                f"{p[1]*100:.2f}%"
            )

            c3.metric(
                "POSITIVO",
                f"{p[2]*100:.2f}%"
            )


with tab2:

    uploaded = st.file_uploader(
        "Sube CSV o Excel",
        type=["csv", "xlsx"]
    )

    if uploaded is not None:

        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        if "text" not in df.columns:

            st.error(
                "El archivo debe contener una columna llamada text."
            )

        else:

            st.write(
                f"Filas cargadas: {len(df):,}"
            )

            if st.button(
                "Analizar dataset"
            ):

                work = df[
                    df["text"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .ne("")
                ].copy()

                pred, probs = predict_batch(
                    work["text"].tolist()
                )

                work["sentiment_predicted"] = [
                    ID2LABEL[int(x)]
                    for x in pred
                ]

                work["confidence"] = probs.max(
                    axis=1
                )

                counts = (
                    work["sentiment_predicted"]
                    .value_counts()
                    .reindex(
                        ["NEGATIVO", "NEUTRAL", "POSITIVO"],
                        fill_value=0
                    )
                )

                pct = counts / len(work) * 100

                st.header(
                    "Distribución de sentimientos"
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "NEGATIVO",
                    f"{pct['NEGATIVO']:.2f}%"
                )

                c2.metric(
                    "NEUTRAL",
                    f"{pct['NEUTRAL']:.2f}%"
                )

                c3.metric(
                    "POSITIVO",
                    f"{pct['POSITIVO']:.2f}%"
                )

                st.bar_chart(
                    pd.DataFrame({
                        "Porcentaje": pct
                    })
                )

                negativas = work[
                    work["sentiment_predicted"]
                    ==
                    "NEGATIVO"
                ].copy()

                filas = []

                for tema, palabras in CATEGORIAS.items():

                    cantidad = int(
                        negativas["text"]
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
                        /
                        len(negativas)
                        *
                        100
                        if len(negativas)
                        else 0
                    )

                    filas.append({
                        "Tema": tema,
                        "Reseñas": cantidad,
                        "% negativas": round(
                            porcentaje,
                            2
                        )
                    })

                temas = (
                    pd.DataFrame(filas)
                    .sort_values(
                        "Reseñas",
                        ascending=False
                    )
                )

                st.header(
                    "Principales temas en reseñas negativas"
                )

                st.dataframe(
                    temas,
                    use_container_width=True
                )

                confianza = (
                    work["confidence"].mean() * 100
                )

                baja = (
                    (work["confidence"] < 0.60)
                    .mean()
                    * 100
                )

                st.header(
                    "Análisis ejecutivo"
                )

                st.write(
                    f"""
Se analizaron **{len(work):,} reseñas**.

El sentimiento predominante fue **{counts.idxmax()}**.

Las reseñas negativas representan
**{pct['NEGATIVO']:.2f}%** del total.

La confianza promedio fue **{confianza:.2f}%**.

El **{baja:.2f}%** de las predicciones tuvo
confianza inferior al 60% y debería revisarse manualmente.

Los temas se identifican mediante reglas de palabras clave;
su presencia indica asociación temática y no causalidad.
"""
                )

                csv = work.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Descargar predicciones",
                    data=csv,
                    file_name="analisis_sentimientos_yelp.csv",
                    mime="text/csv"
                )
