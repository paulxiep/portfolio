import os
import pandas as pd
from functools import reduce
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

ai = 'road'

if __name__ == '__main__'():
    # for trade ai, uncomment the last part
    categorical_columns = list(range(0, 28, 1)) + list(range(47, 47+199, 1)) + [47+199+60] #+ [47+199+61]

    x=[]
    y=[]
    for file in os.listdir(f'intermediate_data/{ai}/x'):
        y_file = file.split('_')
        y_file = y_file[0] + '_y_' + y_file[2]
        try:
            x.append(pd.read_csv(os.path.join(f'intermediate_data/{ai}/x', file), index_col=False, header=None))
            y.append(pd.read_csv(os.path.join(f'intermediate_data/{ai}/y', y_file), index_col=False, header=None))
        except:
            pass

    x = pd.concat(x, axis=0, ignore_index=True)
    y = pd.concat(y, axis=0, ignore_index=True)
    print(len(x.index))
    # the training is at a primitive experimental stage, no need for reproducibility (random_seed)
    # or fancy splitting algorithm yet
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.25)

    x_train = x_train.fillna('none')
    x_val = x_val.fillna('none')

    classifier = CatBoostClassifier(cat_features=categorical_columns)

    classifier.fit(x_train, y_train, eval_set=(x_val, y_val))

    classifier.save_model(f'{ai}_primitive_ml')