FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml ./
COPY gateway ./gateway
RUN pip install --no-cache-dir .

COPY config ./config

ENV GATEWAY_CONFIG=/app/config/gateway.yaml
EXPOSE 8080
CMD ["uvicorn", "--factory", "gateway.main:create_app", "--host", "0.0.0.0", "--port", "8080"]
