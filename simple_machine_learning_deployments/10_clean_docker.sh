docker-compose -f airflow-docker-compose.yaml down
docker-compose -f flask-docker-compose.yaml down
docker image prune -a