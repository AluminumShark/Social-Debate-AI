# Web app image (CPU torch is enough for GNN/RL inference).
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU torch (inference only — small GNN single-node + tiny policy net).
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY docker/requirements-app.txt /tmp/requirements-app.txt
RUN pip install -r /tmp/requirements-app.txt

COPY . /app

EXPOSE 5000
CMD ["gunicorn", "-k", "gthread", "-w", "2", "--threads", "8", "-t", "0", \
     "-b", "0.0.0.0:5000", "wsgi:app"]
