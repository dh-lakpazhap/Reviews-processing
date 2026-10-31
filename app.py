from pathlib import Path

import joblib
import streamlit as st

from core import csv_bytes, predict_reviews

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Анализ отзывов", layout="wide")
st.title("Анализ тональности отзывов")
st.write("Загрузите CSV с английскими отзывами и получите метки тональности.")
st.caption("Обязательный столбец text (content — устаревший вариант для совместимости) · UTF-8 · разделитель: запятая · до 1000 отзывов · до 5 МБ · до 5000 символов в отзыве")


@st.cache_resource
def load_model(path, modified):
    return joblib.load(path)


model_path = ROOT / "sentiment_model.joblib"
if not model_path.exists():
    st.warning("Модель ещё не подключена. Выполните ячейку из EXPORT_MODEL.ipynb в вашем обученном baseline-ноутбуке и положите sentiment_model.joblib рядом с app.py.")
    st.stop()

try:
    model = load_model(str(model_path), model_path.stat().st_mtime_ns)
except Exception:
    st.error("Не удалось загрузить модель. Используйте версии библиотек из requirements.txt, созданного при экспорте модели.")
    st.stop()

uploaded = st.file_uploader("Загрузить CSV", type=["csv"])
if uploaded is None:
    st.info("Для первой проверки загрузите test_data/valid.csv.")
    st.stop()

raw = uploaded.getvalue()
if st.session_state.get("input_bytes") != raw:
    st.session_state.pop("result", None)

if st.button("Анализировать", type="primary"):
    try:
        with st.spinner("Анализируем отзывы..."):
            result = predict_reviews(model, raw)
        st.session_state["result"] = result
        st.session_state["input_bytes"] = raw
    except ValueError as exc:
        st.error(str(exc))
    except Exception:
        st.error("Ошибка прогнозирования. Проверьте модель и установленные зависимости.")

if "result" not in st.session_state:
    st.stop()

result = st.session_state["result"]
negative = result[result["sentiment"] == "negative"]
left, right = st.columns(2)
left.metric("Всего отзывов", len(result))
right.metric("Негативных", len(negative))
only_negative = st.checkbox("Показать только negative")
shown = negative if only_negative else result
st.dataframe(shown, use_container_width=True, hide_index=True)
st.download_button("Скачать все результаты", csv_bytes(result), "all_predictions.csv", "text/csv")
st.download_button("Скачать негативные отзывы", csv_bytes(negative), "negative_reviews.csv", "text/csv")
st.caption("Учебный MVP. Метки получены из звёздных оценок; возможны ошибки. У текущей EXP-02 низкое качество neutral. Качество на новых языках и реальных обращениях не подтверждено.")
