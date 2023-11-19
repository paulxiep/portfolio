#!/bin/bash

python setup.py bdist_wheel
cp dist/restaurant_demo-0.1-py3-none-any.whl flask_dockerfile/python_wheels

docker build flask_dockerfile -t paul-flask:0.0.1

docker-compose -f flask-docker-compose.yaml up -d