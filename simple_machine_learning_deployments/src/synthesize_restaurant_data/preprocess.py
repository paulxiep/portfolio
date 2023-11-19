from functools import lru_cache

import pandas as pd
from sklearn.model_selection import train_test_split

from synthesize_restaurant_data.generate_synthetic_data import synthesize_restaurant_df


def preprocess(df):
    '''
    preprocess for feature list 1
    '''
    return pd.concat([
        pd.concat([
            pd.to_datetime(df['order_acknowledged_at']).dt.hour,
            pd.to_datetime(df['order_acknowledged_at']).dt.weekday,
            pd.to_datetime(df['order_acknowledged_at']).dt.day,
            df['restaurant_id'].map(df.restaurant_id.value_counts())
        ], axis=1, keys=['hour', 'weekday', 'monthday', 'r_counts']),
        df], axis=1
    )[['order_value_gbp',
       'number_of_items',
       'r_counts',
       'monthday',
       'hour',
       'weekday',
       'city',
       'country',
       'type_of_food',
       'restaurant_id',
       ] + ['prep_time_seconds'] * ('prep_time_seconds' in df.columns)]


def post_split_process(df_train, df_test=None, prep_mean=None):
    @lru_cache
    def get_prep_mean():
        if prep_mean is None:
            return df_train.groupby('restaurant_id').prep_time_seconds.mean()
        else:
            return prep_mean

    if df_test is not None:
        return tuple([*get_xy(df_train.merge(get_prep_mean(),
                                             how='left', on='restaurant_id', suffixes=('', '1'))),
                      *get_xy(prepare_test_data(df_test, get_prep_mean())), get_prep_mean()])
    else:
        return tuple([*get_xy(df_train.merge(get_prep_mean(),
                                             how='left', on='restaurant_id', suffixes=('', '1'))), get_prep_mean()])


def get_xy(df, y_col='prep_time_seconds'):
    return df.drop(y_col, axis=1), df[y_col]


def prepare_dummy_data(prep_mean=None):
    '''
    Prepare synthetic data with proper train/test split.
    '''
    return post_split_process(*train_test_split(preprocess(synthesize_restaurant_df()), random_state=0),
                              prep_mean=prep_mean)


def prepare_test_data(data, prep_mean, dropna=True):
    '''
    prepare test data from default format (as obtained form synthesize_restaurant_df)
    into model-ready format, but without removing prep_time_seconds if exists
    '''
    if dropna:
        return preprocess(data).drop(['prep_time_seconds'], axis=1, errors='ignore').merge(prep_mean,
                      how='left', on='restaurant_id', suffixes=('', '1')).dropna()\
                    .rename(columns=lambda x: x if x!='prep_time_seconds' else 'prep_time_seconds1')
    else:
        raise Exception('NotImplementedError')


@lru_cache
def prepare_dummy_model_data():
    '''
    Prepare synthetic data without splitting into train/test,
    used to create dummy model
    '''
    return post_split_process(preprocess(synthesize_restaurant_df()))


if __name__ == '__main__':
    preprocess(synthesize_restaurant_df(10)).to_csv('test_preprocess_df.csv')
