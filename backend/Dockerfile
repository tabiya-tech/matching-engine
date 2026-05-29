FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install CPU-only torch first from the dedicated PyTorch CPU index. The default
# PyPI torch wheel bundles ~2.5GB of NVIDIA CUDA libraries that are useless on
# Cloud Run (no GPU); the CPU wheel is ~180MB. Installing it up front means the
# main install below (incl. sentence-transformers) reuses it instead of pulling CUDA.
RUN pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY resources/ resources/

ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
