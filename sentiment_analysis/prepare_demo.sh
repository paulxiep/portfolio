#!/bin/bash

docker-compose down
docker image rm paul-sentiment:0.0.1
docker build dockerfile -t paul-sentiment:0.0.1
docker-compose up

