#!/bin/bash

docker build airflow_dockerfile -t paul-airflow:0.0.2

docker-compose up -d