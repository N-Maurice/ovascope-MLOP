# OvaScope images. Two targets share one file:
#   docker build --target api -t ovascope-api .
#   docker build --target ui  -t ovascope-ui  .
# Large datasets are NOT baked in: models, database, uploads and the data bundle are mounted volumes.

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OVASCOPE_ROOT=/app
WORKDIR /app

# ---- API: FastAPI + TensorFlow (CPU) for prediction and retraining ----------
FROM base AS api
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt
COPY configs ./configs
COPY src ./src
COPY api ./api
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---- UI: Streamlit only (talks to the API over HTTP, no TensorFlow) ---------
FROM base AS ui
COPY requirements-ui.txt .
RUN pip install -r requirements-ui.txt
COPY configs ./configs
COPY src/__init__.py src/config.py ./src/
COPY app ./app
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
