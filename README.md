# Caso práctico: clasificación de sentimientos en reseñas de Yelp

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

Como extensión práctica se desarrolló una interfaz con Gradio que permite analizar reseñas nuevas individualmente o cargar un conjunto de reseñas para obtener una visión agregada.

En la ejecución sobre 10.000 reseñas se obtuvo:

| Sentimiento predicho | Cantidad | Porcentaje |
|---|---:|---:|
| NEGATIVO | 1.708 | 17,08 % |
| NEUTRAL | 1.848 | 18,48 % |
| POSITIVO | 6.444 | 64,44 % |

La aplicación también permite identificar temas frecuentes dentro de las reseñas clasificadas como negativas y generar un reporte ejecutivo en Excel.

## Estructura

- `notebooks/Caso_Practico_Yelp_Sentimiento.ipynb`: desarrollo académico completo.
- `notebooks/Yelp_Modelo_Final_Aplicacion.ipynb`: aplicación y análisis masivo.

## Reproducibilidad

Semilla principal: `42`.

El dataset original y los pesos entrenados de los modelos no se publican en este repositorio.

El cuaderno académico documenta el preprocesamiento, partición de datos, entrenamiento, evaluación y análisis de errores.

## Nota metodológica

Los resultados corresponden al split y configuración ejecutados en este caso práctico y no implican superioridad universal de un modelo.

El análisis temático de las reseñas negativas de la aplicación utiliza reglas basadas en palabras clave. Por tanto, identifica asociaciones temáticas y no demuestra causalidad.
