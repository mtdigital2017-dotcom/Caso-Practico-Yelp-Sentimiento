# Notebooks/Caso_Practico_Yelp_Sentimiento_FINAL.ipynb

Caso práctico desarrollado para el curso **Natural Language Processing**, dentro de la **Especialización en Inteligencia Artificial para Analítica de Datos**.

**Estudiante:** Julio César Mendoza  
**Docente:** Héctor Manuel Rojas  

## Objetivo

Construir y comparar modelos de clasificación de sentimientos sobre reseñas de Yelp, utilizando exclusivamente el texto de las reseñas como predictor.

Las estrellas se utilizan como etiqueta proxy:

- 1–2 estrellas → NEGATIVO (0)
- 3 estrellas → NEUTRAL (1)
- 4–5 estrellas → POSITIVO (2)

## Modelos desarrollados

Se implementaron y evaluaron:

- Baseline de clase mayoritaria
- TF-IDF + Logistic Regression
- Word2Vec + BiLSTM
- BERT (`google-bert/bert-base-uncased`)

La métrica principal de comparación fue **F1 macro**, debido al desbalance entre clases.

## Resultados en TEST

| Modelo | Accuracy | F1 macro |
|---|---:|---:|
| Clase mayoritaria | 0.6856 | 0.2712 |
| TF-IDF + Logistic Regression | 0.7894 | 0.6871 |
| Word2Vec + BiLSTM | 0.7542 | 0.6326 |
| BERT | **0.8113** | **0.7166** |

En este split y configuración, **BERT obtuvo el mejor F1 macro**.

## Aplicación de analítica de sentimientos

Como extensión práctica se desarrolló una interfaz con Streamlit que permite analizar reseñas nuevas individualmente o cargar un conjunto de reseñas para obtener una visión agregada.

### Aplicación web y estrategias de comunicación

Como extensión práctica del caso se desarrolló una aplicación web con Streamlit que permite utilizar el modelo BERT entrenado para analizar nuevas reseñas, tanto individualmente como de forma masiva.

La aplicación utiliza exclusivamente el texto de la reseña como predictor y clasifica cada comentario en tres categorías:

- NEGATIVO
- NEUTRAL
- POSITIVO

### Funcionalidades

La aplicación permite:

- Clasificar una reseña individual y visualizar las probabilidades estimadas para las tres clases.
- Cargar archivos CSV o Excel con múltiples reseñas.
- Obtener la distribución agregada de sentimientos.
- Identificar temas frecuentes dentro de las reseñas clasificadas como negativas.
- Generar un análisis ejecutivo de los resultados.
- Establecer prioridades de comunicación a partir de los resultados agregados.
- Proponer estrategias, acciones y canales de comunicación para los principales temas negativos.
- Descargar las predicciones obtenidas.

### Aplicación pública

La aplicación se encuentra desplegada en Streamlit Community Cloud:

https://caso-practico-yelp-sentimiento-popb3qpztrhzsggykpkio2.streamlit.app

Por restricciones de recursos del servicio gratuito de despliegue, la aplicación pública utiliza una muestra controlada para el análisis masivo. Esta limitación corresponde únicamente al despliegue y no modifica el entrenamiento, la evaluación ni los resultados académicos obtenidos sobre el conjunto de test.

### Resultados del análisis masivo

En la ejecución previamente realizada sobre 10.000 reseñas se obtuvo:

| Sentimiento predicho | Cantidad | Porcentaje |
|---|---:|---:|
| NEGATIVO | 1.708 | 17,08 % |
| NEUTRAL | 1.848 | 18,48 % |
| POSITIVO | 6.444 | 64,44 % |

Estos resultados corresponden al análisis masivo de nuevas reseñas y no deben confundirse con las métricas de evaluación del modelo sobre el conjunto de test.

### Apoyo a estrategias de comunicación

La aplicación incorpora una capa adicional de analítica para apoyar la interpretación de los resultados. El flujo implementado puede resumirse como:

**Reseñas → BERT → Sentimientos → Temas negativos → Diagnóstico → Prioridades → Estrategias de comunicación → Acciones y canales**

BERT se utiliza exclusivamente para la clasificación del sentimiento. Las estrategias de comunicación no son predicciones generadas por BERT, sino recomendaciones derivadas de reglas analíticas aplicadas sobre los resultados agregados.

Los temas negativos se identifican mediante reglas basadas en palabras clave. Su presencia indica asociación temática y no demuestra causalidad.

De esta manera, la aplicación permite transformar las predicciones de sentimiento en información de apoyo para la toma de decisiones, manteniendo separada la salida del modelo de las recomendaciones posteriores.

## Estructura

- `notebooks/Caso_Practico_Yelp_Sentimiento_FINAL.ipynb`: desarrollo académico completo.
- `notebooks/Yelp_Modelo_Final_Aplicacion.ipynb`: aplicación y análisis masivo.

## Reproducibilidad

Semilla principal: `42`.

El dataset original y los pesos entrenados de los modelos no se publican en este repositorio.

El cuaderno académico documenta el preprocesamiento, partición de datos, entrenamiento, evaluación y análisis de errores.

## Nota metodológica

Los resultados corresponden al split y configuración ejecutados en este caso práctico y no implican superioridad universal de un modelo.

El análisis temático de las reseñas negativas de la aplicación utiliza reglas basadas en palabras clave. Por tanto, identifica asociaciones temáticas y no demuestra causalidad.
