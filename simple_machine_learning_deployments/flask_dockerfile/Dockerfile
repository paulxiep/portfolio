FROM python:3.9.18
RUN pip3 install Flask==2.2.5
RUN pip3 install pandas==2.1.1

RUN mkdir -p /python_wheels
COPY python_wheels python_wheels
RUN python3 -m pip install python_wheels/*

WORKDIR /flask_app
COPY simple_flask .

EXPOSE 5556
CMD [ "flask", "run","--host","0.0.0.0","--port","8081"]