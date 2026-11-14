# Быстрый запуск

## Онлайн

Откройте https://revprocess.streamlit.app/, загрузите test_data/valid.csv из репозитория и нажмите «Анализировать». Ожидается 20 строк, из них 10 negative. Ключ API и обучение не нужны. CSV обрабатывается на сервере, поэтому для демонстрации используются разрешённые обезличенные данные.

## Локально

Скачайте репозиторий или распакуйте его ZIP, откройте корневую папку в терминале. Нужен Python 3.12. Backend проверен на Python 3.12.14. Модель уже находится рядом с app.py.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-backend-verified.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

macOS / Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-backend-verified.txt
.venv/bin/python -m streamlit run app.py
```

Проверенные точные версии относятся к backend, не к облачному серверу Streamlit. Версию Python 3.12 можно проверить командой `python3.12 --version` либо `py -3.12 --version`. При запросе email Streamlit поле можно оставить пустым.

Откройте Local URL из терминала, обычно http://localhost:8501. Загрузите test_data/valid.csv. Проверьте анализ 20 строк и 10 negative, переключение фильтра и обе выгрузки. При включённом фильтре полная выгрузка всё равно содержит 20 строк.

## Тесты

Windows:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/evaluate_ml.py
```

macOS / Linux:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/evaluate_ml.py
```

Ожидается 20 успешных автотестов. ML-команда использует сохранённую модель и локальный датасет, не обучает заново и создаёт `evidence/ml_test_evidence_rerun.json`. Для повторения обучения установите также requirements-ml.txt и откройте baseline.ipynb. Это отдельный сценарий, который не нужен для демонстрации.

## Завершение и ошибки

Ctrl+C останавливает локальный сервер. Если библиотека не найдена, повторите установку через Python созданного окружения. Если модель не найдена, восстановите sentiment_model.joblib из репозитория. CSV должен содержать text либо content, UTF-8 и запятые. Полная диагностика: [Runbook](docs/08_Runbook.md).
