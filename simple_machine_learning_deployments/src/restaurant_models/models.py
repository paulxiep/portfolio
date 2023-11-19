import pickle

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from synthesize_restaurant_data.preprocess import prepare_dummy_model_data
from xgboost import XGBRegressor


def functional_model(model):
    '''
    wrap any machine learning model and override fit method to conform to functional programming style,
    due to compatibility restriction with sklearn, positional argument is not allowed in __init__
    '''

    class FunctionalModel(model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def fit(self, *args, **kwargs):
            super().fit(*args, **kwargs)
            return self

        def load_model(self, file_name):
            super().load_model(file_name)
            return self

        def parent(self):
            return self.__class__.__bases__[0].__name__

        def pickle(self, path):
            self.__class__ = self.__class__.__bases__[0]
            with open(path, 'wb') as f:
                pickle.dump(self, f)

    return FunctionalModel


def get_dummy_models():
    return {
        'rf': functional_model(RandomForestRegressor) \
            (n_estimators=100, min_samples_split=6, max_features='log2'),
        'gbr': functional_model(GradientBoostingRegressor) \
            (n_estimators=100, learning_rate=0.1, min_samples_split=2),
        'cat': functional_model(CatBoostRegressor) \
            (depth=6, iterations=100, learning_rate=0.1, silent=True),
        'xgb': functional_model(XGBRegressor) \
            (max_depth=4, n_estimators=100, learning_rate=0.1)
    }


def train_dummy_data(dummy_models, dummy_data=prepare_dummy_model_data()[:2]):
    return {k: v.fit(*dummy_data) for k, v in dummy_models.items()}


def save_model(model, name='dummy'):
    if isinstance(model, CatBoostRegressor):
        model.save_model(f'{name}_cat.cbm')
    elif isinstance(model, XGBRegressor):
        model.save_model(f'{name}_xg.json')
    else:
        model.pickle(f'{name}_{model.parent()}.pkl')


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def load_models(name='dummy'):
    return {
        'rf': load_pickle(f'{name}_RandomForestRegressor.pkl'),
        'gbr': load_pickle(f'{name}_GradientBoostingRegressor.pkl'),
        'cat': functional_model(CatBoostRegressor)().load_model(f'{name}_cat.cbm'),
        'xgb': functional_model(XGBRegressor)().load_model(f'{name}_xg.json')
    }


if __name__ == '__main__':
    dummy_data = prepare_dummy_model_data()
    [save_model(v) for v in train_dummy_data(get_dummy_models(), dummy_data[:2]).values()]
    dummy_data[2].to_csv('prep_mean.csv')
