# Dashboard service — Python / Streamlit
# Build context: invoice_parse/ root
#   docker build -f services/dashboard/Dockerfile .

FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y libpq5 && rm -rf /var/lib/apt/lists/*

COPY libs/shared-py /tmp/shared-py
RUN pip install --no-cache-dir /tmp/shared-py && rm -rf /tmp/shared-py

COPY services/dashboard/ /app/
RUN pip install --no-cache-dir streamlit

COPY config/ /app/config/

EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
