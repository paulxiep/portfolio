import numpy as np
import pandas as pd


def normal_user_data(mean=0, sd=1, n_user=100, features=tuple(map(lambda x: f'feature_{x}', range(10)))):
    return pd.DataFrame(np.random.normal(mean, sd, (n_user, len(features))), columns=features)
