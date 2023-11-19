import json
import logging
import os

import pandas as pd
from flask import Flask, request
from restaurant_models.models import load_models
from synthesize_restaurant_data.preprocess import prepare_test_data

app = Flask(__name__)

models = load_models('/models/dummy')
prep_mean = pd.read_csv('/models/prep_mean.csv')


@app.post("/predict")
def run_ai():
    if request.headers['aikey'] == os.environ['AIKEY']:
        if request.is_json:
            try:
                data = request.get_json()['json_data']
                logging.info('successfully retrieved data')
                df = pd.DataFrame.from_records(json.loads(data))
                logging.info('successfully converted data to data frame')
                predictions = {k: list(v.predict(prepare_test_data(df, prep_mean) \
                                                 .drop(['prep_time_seconds'], axis=1, errors='ignore')
                                                 ).astype(float))
                               for k, v in models.items()}
                logging.info('successfully ran prediction models')
                return predictions, 201
            except Exception as e:
                return {"error": str(e)}, 400
        else:
            return {"error": "Request must be JSON"}, 415
    else:
        return {'error': 'Wrong key'}, 401


if __name__ == "__main__":
    app.run()
