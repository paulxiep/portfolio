import streamlit as st
from src.constants import card_dict, wonder_names
from itertools import product
from move_caller import BoardData, MoveCaller
from src.model import DQNetwork
import numpy as np

st.title('7 Wonders AI call')
st.subheader('An app for cheating your friends in 7 wonders 2nd edition (base game only)')
st.text('Input built structures, coins, and wonder level\non yours and your immediate neighbor boards\nthen input your card and game stage\nif you\'re playing discarded as Helicarnassus, check playing discarded\nWhen you\'re done, click Make a move!')

col0, col1, col2, col3 = st.tabs(['Card and Play', 'Own Board', 'Left Board', 'Right Board'])

with col0:
    cards = st.multiselect('Cards available', card_dict.keys(), key='card')
    ncard = st.slider('Card Order in Age', 1, 7, 1)
    age = st.slider('Age', 1, 3, 1)
    playing_discard = st.checkbox('Playing Discarded')
    n_discarded = st.slider('Cards in discarded pile', 0, 50, 0)

with col1:
    own_wonder_name, own_wonder_side = st.selectbox('Own Wonder', product(wonder_names, ['A', 'B']))
    own_coins = st.slider('Own Coins', 0, 50, 3)
    own_wonder_built = st.slider('Own Wonder Stages Built', 0, 4, 0)
    own_structures = st.multiselect('Own Structures', card_dict.keys())

with col2:
    left_wonder_name, left_wonder_side = st.selectbox('Left Wonder', product(wonder_names, ['A', 'B']))
    left_coins = st.slider('Left Coins', 0, 50, 3)
    left_wonder_built = st.slider('Left Wonder Stages Built', 0, 4, 0)
    left_structures = st.multiselect('Left Structures', card_dict.keys())

with col3:
    right_wonder_name, right_wonder_side = st.selectbox('Right Wonder', product(wonder_names, ['A', 'B']))
    right_coins = st.slider('Right Coins', 0, 50, 3)
    right_wonder_built = st.slider('Right Wonder Stages Built', 0, 4, 0)
    right_structures = st.multiselect('Right Structures', card_dict.keys())

def call():
    model = DQNetwork(240, hidden_size=512)
    model(np.zeros([1, 380]).astype(float))
    model.load_weights('7_wonders/weights/dqx_7')
    move_caller = MoveCaller(model)
    for player in ['own', 'left', 'right']:
        play_discard = playing_discard if player == 'own' else False
        globals()[f'{player}_board'] = BoardData(
                                            globals()[f'{player}_wonder_name'],
                                            globals()[f'{player}_wonder_side'],
                                            globals()[f'{player}_wonder_built'],
                                            globals()[f'{player}_structures'],
                                            globals()[f'{player}_coins'],
                                            play_discard
                                    )
    return move_caller(cards,
                       own_board.prepare_board(),
                       left_board.prepare_board(),
                       right_board.prepare_board(),
                       ncard,
                       age,
                       n_discarded)

move = st.button('Make a move!')
if move and len(cards) > 0:
    st.subheader(call())
    move = False





