# Processing service — Python
# TODO: implement when service is built

FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "-m", "invoice_processing.worker"]
