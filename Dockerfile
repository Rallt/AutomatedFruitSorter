FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# TensorFlow/OpenCV runtime libraries.
RUN apt-get update && apt-get install --no-install-recommends -y libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY API ./API
COPY core ./core
COPY models ./models
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && pip install .

RUN useradd --create-home --uid 10001 sorter \
    && mkdir -p /app/data/frames \
    && chown -R sorter:sorter /app
USER sorter

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"

CMD ["python", "-m", "uvicorn", "API.SERVE:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
