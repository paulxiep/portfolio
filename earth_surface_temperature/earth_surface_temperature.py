import os
import zipfile
from datetime import date

import kaggle
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import signal

st.set_page_config(layout='wide', page_title='Land Surface Temperature')

st.title('Global Land Surface Temperature')
st.header('Monthly averages since 1890')

st.markdown(
    'from [Berkeley Earth data](https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data)')


@st.cache_data
def load_data():
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('berkeleyearth/climate-change-earth-surface-temperature-data', path='.',
                                      unzip=False)
    with zipfile.ZipFile('climate-change-earth-surface-temperature-data.zip', 'r') as zip_ref:
        df = pd.read_csv(zip_ref.open('GlobalLandTemperaturesByCity.csv'))
    os.remove('climate-change-earth-surface-temperature-data.zip')

    df['dt'] = df['dt'].apply(date.fromisoformat)
    df['Year'] = df['dt'].apply(lambda x: x.year)
    df = df[['Year', 'AverageTemperature', 'City', 'Latitude']]
    df['Latitude'] = df['Latitude'].apply(lambda x: float(x[:-1]) if x[-1] == 'N' else -float(x[:-1]))
    df = df[df['Year'] >= 1890].reset_index().drop('index', axis=1)

    return df


def df_filter_latitude(thres_l, thres_u):
    df = load_data()[load_data()['Latitude'].apply(lambda x: thres_l <= abs(x) <= thres_u)]
    return pd.concat([
        df.groupby(['City', 'Year']).min().reset_index().drop('City', axis=1) \
            .groupby('Year').mean().reset_index() \
            .rename(columns={'AverageTemperature': 'MinAverageTemperature'}) \
            .drop(['Year', 'Latitude'], axis=1),
        df.groupby(['City', 'Year']).max().reset_index().drop('City', axis=1) \
            .groupby('Year').mean().reset_index() \
            .rename(columns={'AverageTemperature': 'MaxAverageTemperature'})
    ], axis=1)


def smooth(data):
    return signal.savgol_filter(data, 21, 1)


def plot(df):
    minc, maxc = st.columns(2)
    with minc:
        st.plotly_chart(px.scatter(df,
                                   x='Year', y='MinAverageTemperature',
                                   title='Minimum yearly temperature, averaged over cities in latitude range',
                                   labels={'MinAverageTemperature': 'Temperature in °C'}  # , color='City'
                                   ).update_traces(
            marker={'size': 4}).add_traces(
            px.line(
                df, x='Year',
                y=smooth(df['MinAverageTemperature']),
                labels={'y': 'smoothed Temperature in °C'}
            ).data
        ).add_hrect(y0=min(smooth(df['MinAverageTemperature'])),
                    y1=max(smooth(df['MinAverageTemperature'])),
                    opacity=0.2, fillcolor='purple', line_width=0),
                        use_container_width=True)
    with maxc:
        st.plotly_chart(px.scatter(df,
                                   x='Year', y='MaxAverageTemperature',
                                   title='Maximum yearly temperature, averaged over cities in latitude range',
                                   labels={'MaxAverageTemperature': 'Temperature in °C'}  # , color='City'
                                   ).update_traces(
            marker={'size': 4}).add_traces(
            px.line(
                df, x='Year',
                y=smooth(df['MaxAverageTemperature']),
                labels={'y': 'smoothed Temperature in °C'}
            ).data
        ).add_hrect(y0=min(smooth(df['MaxAverageTemperature'])),
                    y1=max(smooth(df['MaxAverageTemperature'])),
                    opacity=0.2, fillcolor='purple', line_width=0),
                        use_container_width=True)


with st.expander('Min-Max temperature, averaged over cities in selected latitude range'):
    st.text('Equivalent Southern latitudes included')
    threslc, thresuc = st.columns(2)
    with threslc:
        thres_l = st.slider('min latitude', 0, 60, value=0, step=5)
    with thresuc:
        thres_u = st.slider('max latitude', thres_l + 10, 70,
                            value=max(thres_l + 10, st.session_state.get('thres_u', 70)), step=5)
    st.session_state['thres_u'] = thres_u
    plot(df_filter_latitude(thres_l, thres_u))
