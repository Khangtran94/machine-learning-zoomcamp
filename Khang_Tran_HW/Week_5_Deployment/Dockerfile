FROM agrigorev/zoomcamp-model:2025

WORKDIR /code

COPY pyproject.toml .

RUN pip install --no-cache-dir scikit-learn uvicorn fastapi requests

COPY . .

EXPOSE 8000

CMD ["uvicorn", "web_model:app", "--host", "0.0.0.0", "--port", "8000"] 