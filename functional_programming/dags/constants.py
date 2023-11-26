'''
Set the 'CONSTANTS' here.
Metric minimum threshold set here so it can be changed without restarting Airflow
'''
METRIC_MINIMUM_THRESHOLD = 0.8
DATA_MODE = 'synthesize'
TRAINING_SIZE = 100
EVALUATION_SIZE = 20
DAG_DEFAULT_ARGS = {
    "email": [],
    "email_on_failure": False
}