#!/bin/bash

docker build airflow_dockerfile -t paul-airflow:0.0.1
docker-compose -f airflow-docker-compose.yaml up -d