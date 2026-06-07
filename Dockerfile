FROM python:3.11.15-slim

WORKDIR /app

# Build deps for llama-cpp-python (CPU-only on Linux x86)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/
COPY configs/ configs/
COPY data/ data/
COPY scripts/ scripts/

RUN pip install --no-cache-dir -e .

VOLUME ["/app/results", "/root/.cache/tiny-llm-edge-bench"]

ENTRYPOINT ["python", "scripts/run_bench.py"]
