FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (better layer caching).
COPY requirements.txt requirements-db.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-db.txt

# App code.
COPY app ./app
COPY pyproject.toml README.md Info.md ./

EXPOSE 8000

ENV ENVIRONMENT=production \
    LOG_LEVEL=INFO

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
