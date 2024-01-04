FROM nvidia/cuda:11.6.1-cudnn8-runtime-ubuntu20.04

# Install necessary libs
RUN apt update && apt install -y python3-pip python3.8-dev && rm -rf /var/lib/apt/lists/*

# Install python3.8
RUN apt install -y python3.8
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2

# Install dependencies
RUN python3 -m pip install pip
RUN python3 -m pip install --upgrade pip
RUN pip3 install pandas==2.0.3\
                    numpy==1.24.4\
                    pythainlp==4.0.2\
                    scikit-learn==1.3.2\
                    tensorflow==2.10.1\
                    flask==2.2.5

RUN mkdir -p /python_wheels
COPY python_wheels python_wheels
RUN python3 -m pip install python_wheels/*

WORKDIR /flask_app
COPY simple_flask .

EXPOSE 5556
CMD [ "flask", "run","--host","0.0.0.0","--port","8081"]

