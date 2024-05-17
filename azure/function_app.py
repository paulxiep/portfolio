import json
import os
import pickle
import re

import azure.functions as func
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request
from pythainlp import word_tokenize

flask_app = Flask(__name__)

with open('./models/logistic_regression.pkl', 'rb') as f:
    logistic_regression_model = pickle.load(f)
with open('./models/naive_bayes.pkl', 'rb') as f:
    naive_bayes_model = pickle.load(f)
with open('./models/sklearn_vectorizer.pkl', 'rb') as f:
    sklearn_vectorizer = pickle.load(f)
lstm_model = tf.keras.models.load_model('./models/lstm.h5', compile=False)
lstm_vectorizer = tf.keras.models.load_model('./models/tf_vectorizer')
transformers_vectorizer = tf.keras.models.load_model('./models/tf_vectorizer_t')
transformers_model = tf.keras.models.load_model('./models/transformers', compile=False)

batch_size = 32


def filter_thai(text):
    '''
    basically, filter out special characters
    '''
    pattern = re.compile(r"[^\u0E00-\u0E7F ]|^'|'$|''")
    char_to_remove = re.findall(pattern, text)
    list_with_char_removed = [char for char in text if not char in char_to_remove]
    return ''.join(list_with_char_removed)


def tokenize_text(x):
    return ' '.join(list(filter(lambda y: y.replace(' ', ''), word_tokenize(filter_thai(x)))))


@flask_app.post("/predict")
def run_ai():
    if request.headers['aikey'] == os.environ['AIKEY']:
        if request.is_json:
            try:
                json_body = request.get_json()
                data = json_body['json_data']
                model_choice = json_body['model_choice']
                # logging.info('successfully retrieved data')
                data = pd.DataFrame.from_dict(json.loads(data))
                # data was reordered by string indexing as it was serialized and sent
                data = data.loc[list(map(str, range(len(data.index))))]
                # logging.info('successfully converted data to data frame')
                if model_choice != 'Vote':
                    predictions = {model_choice: None}
                else:
                    predictions = {'Transformers': None, 'LSTM': None, 'Logistic Regression': None, 'Naive Bayes': None}
                data['0'] = data['0'].apply(filter_thai)
                if 'Logistic Regression' in predictions.keys():
                    predictions['Logistic Regression'] = logistic_regression_model.predict_proba(
                        sklearn_vectorizer.transform(data['0']).toarray()
                    ).tolist()
                if 'Naive Bayes' in predictions.keys():
                    predictions['Naive Bayes'] = naive_bayes_model.predict_proba(
                        sklearn_vectorizer.transform(data['0']).toarray()
                    ).tolist()
                data['0'] = data['0'].apply(tokenize_text)
                if 'LSTM' in predictions.keys():
                    predictions['LSTM'] = np.concatenate([lstm_model(lstm_vectorizer(
                        data.iloc[(batch_size * i):min((batch_size * (i + 1)), len(data.index))])).numpy() for i in
                                                          range(1 + len(data.index) // batch_size)], axis=0).tolist()
                if 'Transformers' in predictions.keys():
                    predictions['Transformers'] = np.concatenate([transformers_model(transformers_vectorizer(
                        data.iloc[(batch_size * i):min((batch_size * (i + 1)), len(data.index))])).numpy() for i in
                                                                  range(1 + len(data.index) // batch_size)],
                                                                 axis=0).tolist()
                # logging.info('successfully ran prediction models')
                return {'predictions': predictions,
                        'message': tf.config.list_physical_devices('GPU')
                        }, 201
            except Exception as e:
                return {"error": str(e)}, 400
        else:
            return {"error": "Request must be JSON"}, 415
    else:
        return {'error': 'Wrong key'}, 401


app = func.WsgiFunctionApp(app=flask_app.wsgi_app,
                           http_auth_level=func.AuthLevel.ANONYMOUS)
