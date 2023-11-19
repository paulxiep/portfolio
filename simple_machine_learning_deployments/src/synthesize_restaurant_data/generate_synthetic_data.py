import datetime
import random
import os

import numpy as np
import pandas as pd
from scipy import stats

from synthesize_restaurant_data.synthetic_data_parameters import RANGE, N_DATA, COLUMNS


def skew_norm_pdf(e=0, w=1, a=-1):
    '''
    modified from simple function in Stackoverflow into higher order function
    '''

    def get_value(x):
        t = (x - e) / w
        return 2.0 * w * stats.norm.pdf(t) * stats.norm.cdf(a * t)

    return get_value


def synthesize(range_value, n_data=N_DATA):
    '''
    the goal is not to accurately mimic data in jupyter notebook,
    nor is it to have any correlation between x and y in the synthesized data
    '''
    if range_value is None:
        return np.vectorize(lambda x: datetime.datetime.fromtimestamp(x - random.randrange(0, 60 * 60 * 24 * 365))) \
            (np.zeros(n_data) + datetime.datetime.now().timestamp())
    elif isinstance(range_value, tuple):
        return np.vectorize(skew_norm_pdf(range_value[0], range_value[1]))\
                    (np.random.random(n_data) * (range_value[0] + 2 * range_value[1]))
    elif isinstance(range_value, int):
        return np.random.randint(0, range_value, n_data)
    else:
        raise Exception('Unknown range value type')


def synthesize_restaurant_df(n_data=N_DATA):
    return pd.DataFrame.from_dict(dict(zip(COLUMNS, map(lambda x: synthesize(x, n_data), RANGE))))


# if __name__ == '__main__':
#     synthesize_restaurant_df(10).to_csv('test_synthesize_df.csv', index=False)
