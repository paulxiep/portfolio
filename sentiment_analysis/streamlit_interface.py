import environ
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

from src.paul_sentiment_analysis.prepare_data import prepare_train_test

st.set_page_config(
    page_title='Sentiment Analysis',
)
st.title('Sentiment Analysis')
st.header('Sentiment Analysis trained on Thai google map reviews data of 11 venue types in 12 provinces in Thailand')
environ.Env.read_env()


def call_model(data, model_choice):
    st.markdown('ran flask api')
    headers = {
        'aikey': environ.Env()('AIKEY')
    }
    json_body = {
        'json_data': data.to_json(),
        'model_choice': model_choice
    }
    host = f"http://localhost:{environ.Env()('FLASK_PORT')}"
    response = requests.post(f"{host}/predict",
                             headers=headers,
                             json=json_body).json()
    return response


def predict_input(review_input, model_choice):
    if isinstance(review_input, str):
        # if 'csv' not in review_input:
        data = pd.DataFrame([[review_input]])
        labels = None
        # else:
        #     data = pd.read_csv(review_input)
        #     data.columns = [i for i in range(len(data.columns))]
        #     if 1 in data.columns:
        #         labels = data[1].apply(lambda x: {0: 'neg', 1: 'neu', 2: 'pos'}.get(x, x))
        #     else:
        #         labels = None
        #     data = data[[0]]

    elif isinstance(review_input, pd.DataFrame):
        data = review_input
        data.columns = [i for i in range(len(data.columns))]
        data = data.reset_index()[[0, 1]]
        if 1 in data.columns:
            labels = data[1].apply(lambda x: {0: 'neg', 1: 'neu', 2: 'pos'}.get(x, x))
        else:
            labels = None
        data = data[[0]]

    else:
        raise NotImplementedError('unexpected input for predict_input')

    return call_model(data, model_choice), data, labels


def display_predictions(predictions, data, labels):
    predictions, message = predictions['predictions'], predictions['message']
    probabilities = np.array([np.array(pd.DataFrame.from_dict(predictions[key])) for key in predictions.keys()])
    probabilities = probabilities.mean(axis=0)
    probabilities = pd.DataFrame(probabilities).rename(columns={
        0: 'score_negative',
        1: 'score_neutral',
        2: 'score_positive'
    })
    probabilities['prediction'] = np.vectorize(lambda x: {0: 'neg', 1: 'neu', 2: 'pos'}[x]
                                               )(np.argmax(np.array(probabilities), axis=1))
    if labels is not None:
        probabilities['label'] = labels
        st.write(f"accuracy: {accuracy_score(probabilities['prediction'], probabilities['label'])}")
        st.write(f"f1: {f1_score(probabilities['prediction'], probabilities['label'], average='weighted')}")
        confusion = confusion_matrix(probabilities['prediction'], probabilities['label'])
        st.pyplot(sns.heatmap(confusion, annot=True, fmt='d',
                              xticklabels=['neg', 'neu', 'pos'], yticklabels=['neg', 'neu', 'pos']).get_figure())
    probabilities['text'] = data

    st.dataframe(probabilities)
    with st.expander('GPU check'):
        st.write('GPU check', message)


model_choice = st.radio('choose model', options=['LSTM',
                                                 'Logistic Regression',
                                                 'Naive Bayes',
                                                 'Vote'])
tab1, tab2, tab3 = st.tabs(['predict text input', 'predict csv input', 'predict validation set'])

with tab1:
    text_input = st.text_input('Input review text')
    predict_button_tab1 = st.button('predict text input')
    if predict_button_tab1:
        display_predictions(*predict_input(text_input, model_choice))

with tab2:
    uploaded_file = st.file_uploader('Choose a CSV file')
    predict_button_tab2 = st.button('predict csv input')
    if uploaded_file and predict_button_tab2:
        display_predictions(*predict_input(
            pd.read_csv(uploaded_file),
            model_choice)
        )

with tab3:
    predict_button_tab3 = st.button('predict validation set')
    if predict_button_tab3:
        _, test_set, _ = prepare_train_test()
        display_predictions(*predict_input(
            test_set[['review', 'rating']],
            model_choice)
        )
