import json
import logging
import os
import pickle

import pandas as pd
import tensorflow as tf
from flask import Flask, request
from paul_sentiment_analysis.tokenize_data import tokenize_text, filter_thai

app = Flask(__name__)
logging.getLogger().setLevel(logging.DEBUG)

with open('/models/logistic_regression.pkl', 'rb') as f:
    logistic_regression_model = pickle.load(f)
with open('/models/naive_bayes.pkl', 'rb') as f:
    naive_bayes_model = pickle.load(f)
with open('/models/sklearn_vectorizer.pkl', 'rb') as f:
    sklearn_vectorizer = pickle.load(f)
lstm_model = tf.keras.models.load_model('/models/lstm.h5', compile=False)
lstm_vectorizer = tf.keras.models.load_model('/models/tf_vectorizer')


@app.post("/predict")
def run_ai():
    if request.headers['aikey'] == os.environ['AIKEY']:
        if request.is_json:
            try:
                json_body = request.get_json()
                data = json_body['json_data']
                model_choice = json_body['model_choice']
                logging.info('successfully retrieved data')
                data = pd.DataFrame.from_dict(json.loads(data))
                # data was reordered by string indexing as it was serialized and sent
                data = data.loc[list(map(str, range(len(data.index))))]
                logging.info('successfully converted data to data frame')
                if model_choice != 'Vote':
                    predictions = {model_choice: None}
                else:
                    predictions = {'LSTM': None, 'Logistic Regression': None, 'Naive Bayes': None}
                data['0'] = data['0'].apply(filter_thai)
                if 'Logistic Regression' in predictions.keys():
                    predictions['Logistic Regression'] = logistic_regression_model.predict_proba(
                        sklearn_vectorizer.transform(data['0']).toarray()
                    ).tolist()
                if 'Naive Bayes' in predictions.keys():
                    predictions['Naive Bayes'] = naive_bayes_model.predict_proba(
                        sklearn_vectorizer.transform(data['0']).toarray()
                    ).tolist()
                if 'LSTM' in predictions.keys():
                    data['0'] = data['0'].apply(tokenize_text)
                    predictions['LSTM'] = lstm_model(lstm_vectorizer(data)).numpy().tolist()
                logging.info('successfully ran prediction models')
                return {'predictions': predictions,
                        'message': tf.config.list_physical_devices('GPU')
                        }, 201
            except Exception as e:
                return {"error": str(e)}, 400
        else:
            return {"error": "Request must be JSON"}, 415
    else:
        return {'error': 'Wrong key'}, 401


if __name__ == "__main__":
    app.run()
