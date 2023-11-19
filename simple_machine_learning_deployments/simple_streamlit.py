import os
import sys

import environ
import pandas as pd
import requests
import streamlit as st

if os.path.join(os.getcwd(), 'src') not in sys.path:
    sys.path.append(os.path.join(os.getcwd(), 'src'))
from restaurant_models.models import load_models
from synthesize_restaurant_data.preprocess import prepare_test_data
from synthesize_restaurant_data.synthetic_data_parameters import COLUMNS, DTYPES

st.set_page_config(
    page_title='Restaurant Regression',
)
st.title('Restaurant Data Regression')

environ.Env.read_env()


def load_csv(test_data):
    test_data.seek(0)
    return pd.read_csv(test_data)


def predict(mode):
    def predict_local(test_data):
        st.markdown('called local module')
        models = load_models('models/dummy')
        return {k: v.predict(prepare_test_data(
            test_data,
            pd.read_csv('models/prep_mean.csv')
        ).drop(['prep_time_seconds'], axis=1, errors='ignore')
                             )
                for k, v in models.items()}

    def predict_flask(test_data):
        st.markdown('ran flask api')
        headers = {
            'aikey': environ.Env()('AIKEY')
        }
        json_body = {
            'json_data': test_data.to_json()
        }
        host = f"http://localhost:{environ.Env()('FLASK_PORT')}"
        response = requests.post(f"{host}/predict",
                                 headers=headers,
                                 json=json_body).json()

        return response

    if mode == 'call local module':
        return predict_local
    elif mode == 'call flask API':
        return predict_flask


def display_predictions(predictions):
    tabs = st.tabs(predictions.keys())
    for i, tab in enumerate(tabs):
        with tab:
            st.dataframe(pd.DataFrame(list(predictions.values())[i], columns=['predicted_prep_time']))
    st.markdown('Using downloader for model results instead of UI display is a simple adjustment.')


st.json(dict(zip(COLUMNS, DTYPES)))
run_mode = st.radio('Choose run mode', options=['call local module', 'call flask API'])
test_data = st.file_uploader(
    'Upload data that conforms to data dictionary above,\ncategories should be encoded as numbers\nand should not exceed number of categories of data in jupyter notebook.\nNumbers are likewise only tested with numbers in range of sample data.')

if test_data is not None:
    display_predictions(predict(run_mode)(load_csv(test_data)))
