import csv
import io

import pandas as pd

MAX_BYTES = 5_000_000
MAX_ROWS = 1000
MAX_CHARS = 5000
LABELS = {"negative", "neutral", "positive"}


def read_reviews(raw):
    if len(raw) > MAX_BYTES:
        raise ValueError("Размер CSV превышает 5 МБ (5 000 000 байт).")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Нужен CSV в кодировке UTF-8.") from exc
    csv.field_size_limit(MAX_BYTES)
    try:
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as exc:
        raise ValueError("CSV повреждён: проверьте кавычки и разделители.") from exc
    if not rows or not rows[0]:
        raise ValueError("CSV пуст.")
    header, data = rows[0], rows[1:]
    if len(header) != len(set(header)) or any(not name.strip() for name in header):
        raise ValueError("Названия столбцов должны быть непустыми и уникальными.")
    if "text" not in header and "content" not in header:
        raise ValueError("В CSV нужен столбец text (также поддерживается content).")
    if not data:
        raise ValueError("В CSV нет отзывов.")
    if len(data) > MAX_ROWS:
        raise ValueError("Допускается не более 1000 отзывов.")
    if any(len(row) != len(header) for row in data):
        raise ValueError("CSV повреждён: неверное число полей или пустая строка.")
    frame = pd.DataFrame(data, columns=header)
    column = "text" if "text" in header else "content"
    texts = frame[column].str.strip()
    if texts.eq("").any():
        raise ValueError("Есть пустые отзывы. Заполните или удалите их.")
    if texts.str.len().gt(MAX_CHARS).any():
        raise ValueError("Отзыв не должен превышать 5000 символов.")
    return frame, texts


def predict_reviews(model, raw):
    frame, texts = read_reviews(raw)
    predictions = list(model.predict(texts))
    if len(predictions) != len(frame) or not set(predictions).issubset(LABELS):
        raise ValueError("Модель вернула некорректные метки.")
    # Выход по техническому заданию: row_id, text, sentiment.
    source_column = "text" if "text" in frame.columns else "content"
    result = pd.DataFrame({
        "row_id": range(1, len(frame) + 1),
        "text": frame[source_column].tolist(),
        "sentiment": predictions,
    })
    return result


def csv_bytes(frame):
    return frame.to_csv(index=False).encode("utf-8-sig")
