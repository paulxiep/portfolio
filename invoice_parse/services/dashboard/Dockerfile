# Dashboard service — Python / Streamlit
# TODO: implement when service is built

FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501"]
