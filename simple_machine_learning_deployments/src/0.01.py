from restaurant_models.models import save_model, train_dummy_data, get_dummy_models
from synthesize_restaurant_data.preprocess import prepare_dummy_model_data

dummy_data = prepare_dummy_model_data()
[save_model(v, name='../models/dummy') for v in train_dummy_data(get_dummy_models(), dummy_data[:2]).values()]
dummy_data[2].to_csv('../models/prep_mean.csv')