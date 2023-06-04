import os
import shutil
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

st.markdown('from [Berkeley Earth data](https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data)')


def load_data():
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('berkeleyearth/climate-change-earth-surface-temperature-data', path='.',
                                      unzip=False)
    with zipfile.ZipFile('climate-change-earth-surface-temperature-data.zip', 'r') as zip_ref:
        df = pd.read_csv(zip_ref.open('GlobalLandTemperaturesByCity.csv'))

    df['dt'] = df['dt'].apply(date.fromisoformat)
    df['Year'] = df['dt'].apply(lambda x: x.year)
    df = df[['Year', 'dt', 'AverageTemperature', 'City', 'Latitude', 'Longitude']]
    df['AdjustedTemperature'] = (df['AverageTemperature'] * 9 / 5) + 32 + 45
    df['Latitude'] = df['Latitude'].apply(lambda x: float(x[:-1]) if x[-1] == 'N' else -float(x[:-1]))
    df['Longitude'] = df['Longitude'].apply(lambda x: float(x[:-1]) if x[-1] == 'E' else -float(x[:-1]))
    df = df[df['Year'] >= 1890].reset_index().drop('index', axis=1)

    return df


@st.cache_data
def df_filter_latitude(thres_l, thres_u):
    return load_data()[load_data()['Latitude'].apply(lambda x: thres_l <= abs(x) <= thres_u)]


def min_temp(thres_l, thres_u):
    return df_filter_latitude(thres_l, thres_u).drop('dt', axis=1) \
        .groupby(['City', 'Year']).min().reset_index().drop('City', axis=1).groupby('Year').mean().reset_index()


def max_temp(thres_l, thres_u):
    return df_filter_latitude(thres_l, thres_u).drop('dt', axis=1) \
        .groupby(['City', 'Year']).max().reset_index().drop('City', axis=1).groupby('Year').mean().reset_index()


@st.cache_data
def smooth(data):
    return signal.savgol_filter(data, 21, 1)


with st.expander('Min-Max temperature, averaged over cities in selected latitude range'):
    st.text('Equivalent Southern latitudes included')
    threslc, thresuc = st.columns(2)
    with threslc:
        thres_l = st.slider('min latitude', 0, 60, value=0, step=5)
    with thresuc:
        thres_u = st.slider('max latitude', thres_l + 10, 70,
                            value=max(thres_l + 10, st.session_state.get('thres_u', 70)), step=5)
    st.session_state['thres_u'] = thres_u
    minc, maxc = st.columns(2)
    with minc:
        st.plotly_chart(px.scatter(min_temp(thres_l, thres_u),
                                   x='Year', y='AverageTemperature',
                                   title='Minimum yearly temperature, averaged over cities in latitude range',
                                   labels={'AverageTemperature': 'Temperature in °C'}  # , color='City'
                                   ).update_traces(
            marker={'size': 4}).add_traces(
            px.line(
                min_temp(thres_l, thres_u), x='Year',
                y=smooth(min_temp(thres_l, thres_u)['AverageTemperature']),
            ).data
        ).add_hrect(y0=min(smooth(min_temp(thres_l, thres_u)['AverageTemperature'])),
                    y1=max(smooth(min_temp(thres_l, thres_u)['AverageTemperature'])),
                    opacity=0.2, fillcolor='purple', line_width=0),
                        use_container_width=True)
    with maxc:
        st.plotly_chart(px.scatter(max_temp(thres_l, thres_u),
                                   x='Year', y='AverageTemperature',
                                   title='Maximum yearly temperature, averaged over cities in latitude range',
                                   labels={'AverageTemperature': 'Temperature in °C'}  # , color='City'
                                   ).update_traces(
            marker={'size': 4}).add_traces(
            px.line(
                max_temp(thres_l, thres_u), x='Year',
                y=smooth(max_temp(thres_l, thres_u)['AverageTemperature']),
            ).data
        ).add_hrect(y0=min(smooth(max_temp(thres_l, thres_u)['AverageTemperature'])),
                    y1=max(smooth(max_temp(thres_l, thres_u)['AverageTemperature'])),
                    opacity=0.2, fillcolor='purple', line_width=0),
                        use_container_width=True)

with st.expander('On the map over time'):
    st.plotly_chart(
        px.scatter_geo(load_data()[load_data()['Year'].apply(lambda x: x % 10 == 0)], lat='Latitude', lon='Longitude',
                       size="AdjustedTemperature",
                       size_max=6,
                       animation_frame='dt', labels={'AverageTemperature': 'Monthly Average Temperature'},
                       range_color=[-45, 45],
                       color_continuous_scale='Cividis',
                       color="AverageTemperature",
                       hover_data={'AdjustedTemperature': False,
                                   'dt': False,
                                   'City': True,
                                   'AverageTemperature': True,
                                   'Latitude': True,
                                   'Longitude': True}, height=600
                       ), use_container_width=True)
