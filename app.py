from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Анализ отзывов", layout="wide")

st.title("Анализ тональности отзывов")
st.write(
    "Загрузите CSV в кодировке UTF-8 со столбцом content. "
    "Разделитель — запятая."
)
st.caption(
    "Учебная модель для отзывов о телеком-приложениях. "
    "Качество распознавания нейтральных отзывов пока низкое."
)


@st.cache_resource
def load_model():
    path = Path(__file__).parent / "sentiment_model.joblib"
    return joblib.load(path)


model_path = Path(__file__).parent / "sentiment_model.joblib"

if not model_path.exists():
    st.error("Рядом с app.py не найден sentiment_model.joblib.")
    st.stop()

model = load_model()

uploaded_file = st.file_uploader("Выберите CSV", type=["csv"])

if uploaded_file is None:
    st.stop()

try:
    reviews = pd.read_csv(uploaded_file, encoding="utf-8-sig")
except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
    st.error("Не удалось прочитать CSV. Проверьте формат и кодировку.")
    st.stop()

if "content" not in reviews.columns:
    st.error("В файле должен быть столбец content.")
    st.stop()

if reviews.empty:
    st.error("В файле нет отзывов.")
    st.stop()

texts = reviews["content"].astype("string").fillna("").str.strip()

if texts.eq("").any():
    st.error("Есть пустые отзывы. Заполните или удалите их.")
    st.stop()

with st.spinner("Анализируем отзывы..."):
    result = reviews.copy()
    result["sentiment"] = model.predict(texts)

negative = result[result["sentiment"] == "negative"]

col1, col2 = st.columns(2)
col1.metric("Всего отзывов", len(result))
col2.metric("Негативных", len(negative))

only_negative = st.checkbox("Показать только негативные")
st.dataframe(
    negative if only_negative else result,
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Скачать все результаты",
    data=result.to_csv(index=False).encode("utf-8-sig"),
    file_name="all_predictions.csv",
    mime="text/csv",
)

st.download_button(
    "Скачать негативные отзывы",
    data=negative.to_csv(index=False).encode("utf-8-sig"),
    file_name="negative_reviews.csv",
    mime="text/csv",
)