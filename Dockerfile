FROM python:3.11 AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1'
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

FROM base AS test

ENV RUN_STRESS_TEST=false

RUN pytest tests -v --disable-warnings --maxfail=1



FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
