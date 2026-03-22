# Processing service — Python
# Build context must be invoice_parse/ root (for shared lib access)
#   docker build -f services/processing/Dockerfile .

FROM python:3.12-slim

# System deps for OpenCV (PaddleOCR dependency)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install shared lib first (changes less often)
COPY libs/shared-py /tmp/shared-py
RUN pip install --no-cache-dir /tmp/shared-py && rm -rf /tmp/shared-py

# Install processing service deps
COPY services/processing/pyproject.toml /app/
COPY services/processing/invoice_processing/ /app/invoice_processing/
RUN pip install --no-cache-dir . && pip install --no-cache-dir "paddlex[ocr]"

# Copy config
COPY config/ /app/config/

CMD ["python", "-m", "invoice_processing.worker"]
