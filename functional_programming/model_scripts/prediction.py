import pandas as pd
import random


def predict(model, data):
    '''
    placeholder for real prediction script
    '''
    return pd.Series(random.choices(range(1000), k=20), name='prediction_results')