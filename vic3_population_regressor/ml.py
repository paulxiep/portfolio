import os
import re
import pandas as pd
import numpy as np
from functools import reduce
import json
from math import e
from data_preparation import get_map, get_provinces_terrain, get_states, \
                            get_state_pops, get_state_resources, get_state_buildings, get_state_terrain

data_dir = '/Games/Steam/steamapps/common/Victoria 3/game'
map_data_file = os.path.join(data_dir, 'map_data/provinces.png')
resource_data_dir = os.path.join(data_dir, 'map_data/state_regions')
pop_data_dir = os.path.join(data_dir, 'common/history/pops')
state_data_file = os.path.join(data_dir, 'common/history/states/00_states.txt')
terrain_file = os.path.join(data_dir, 'map_data/province_terrains.txt')
building_dir = os.path.join(data_dir, 'common/history/buildings')

def combine_state_data(sets):
    out = {}
    for key in sets[0].keys():
        out[key] = reduce(lambda x, y: {**x, **y}, map(lambda z: z[key], sets))
    return out

def cross_features(df, set1, set2):
    from itertools import product
    to_add = []
    for pair in product(set1, set2):
        to_add.append(df[pair[0]] * df[pair[1]])
    return pd.concat(to_add, axis='columns', keys=list(map(lambda x: f'{x[0]}_{x[1]}', product(set1, set2))))

def get_one_hots(sets, df):
    def safe_log(array):
        return array.map(lambda x: np.math.log(x*e) if x > 0 else 0)

    def one_hot_terrain(extra=None):
        terrains = list(df.iloc[0]['terrain'].keys())
        terrains.remove('coast')
        if extra is not None:
            terrains = list(map(lambda x: x+f'_{extra}', terrains))
        to_add = []
        for col in terrains:
            if extra is None:
                to_add.append(df['terrain'].map(lambda x: x.get(col, 0)))
            elif extra == 'log':
                to_add.append(safe_log(df['terrain'].map(lambda x: x.get(col, 0))))
            elif extra == 'square':
                to_add.append(df['terrain'].map(lambda x: x.get(col, 0) ** 2))
        return pd.concat(to_add, axis='columns', keys=terrains)

    def one_hot_coast(extra=None):
        # print(df.iloc[0])
        if extra is None:
            return pd.DataFrame({'coast': df['terrain'].map(lambda x: x.get('coast', 0))})
        elif extra == 'log':
            return pd.DataFrame({'coast_log': safe_log(df['terrain'].map(lambda x: x.get('coast', 0)))})
        elif extra == 'square':
            return pd.DataFrame({'coast_square': df['terrain'].map(lambda x: x.get('coast', 0) ** 2)})

    def one_hot_buildings():
        def buildings_in_file(file):
            return list(map(lambda x: x[10:-3], re.findall("\nbuilding_.* = ", '\n' + open(data_dir + '/common/buildings/' + file, 'r').read())))
        building_files = list \
            (filter(lambda x: '.txt' in x and 'monument' not in x, os.listdir(data_dir + '/common/buildings')))
        all_buildings = reduce(list.__add__, map(buildings_in_file, building_files))
        to_add = []
        for col in all_buildings:
            to_add.append(df['buildings'].map(lambda x: x.get(col, 0)))
        return pd.concat(to_add, axis='columns', keys=all_buildings)

    def one_hot_subsistence(extra=None):
        subsist = ['building_subsistence_farms', 'building_subsistence_orchards', 'building_subsistence_pastures']
        if extra is not None:
            subsist = list(map(lambda x: x+f'_{extra}', subsist))
        to_add = []
        for col in subsist:
            if extra is None:
                to_add.append(df['subsistence_building'].map(lambda x: col==x)*df['arable_land'])
            elif extra == 'log':
                to_add.append(safe_log(df['subsistence_building'].map(lambda x: col==x)*df['arable_land']))
            elif extra == 'square':
                to_add.append((df['subsistence_building'].map(lambda x: col==x)*df['arable_land'])**2)
            # to_add.append(df['subsistence_building'].map(lambda x: col==x) * df['arable_land'])
        return pd.concat(to_add, axis='columns', keys=subsist)

    def one_hot_farms(extra=None):
        farms = ['bg_rye_farms', 'bg_wheat_farms',
                 'bg_rice_farms', 'bg_maize_farms', 'bg_millet_farms',
                 'bg_livestock_ranches']
        if extra is not None:
            farms = list(map(lambda x: x+f'_{extra}', farms))
        to_add = []
        for col in farms:
            if extra is None:
                to_add.append(df['arable_resources'].map(lambda x: col==x)*df['arable_land'])
            elif extra == 'log':
                to_add.append(safe_log(df['arable_resources'].map(lambda x: col==x)*df['arable_land']))
            elif extra == 'square':
                to_add.append((df['arable_resources'].map(lambda x: col==x)*df['arable_land'])**2)
            # to_add.append(df['arable_resources'].map(lambda x: col in x) * df['arable_land'])
        return pd.concat(to_add, axis='columns', keys=farms)

    def one_hot_non_food():
        plantations = ['bg_logging', 'bg_coffee_plantations',
                       'bg_cotton_plantations', 'bg_dye_plantations', 'bg_opium_plantations',
                       'bg_tea_plantations', 'bg_tobacco_plantations', 'bg_sugar_plantations',
                       'bg_banana_plantations', 'bg_silk_plantations']
        to_add = []
        for col in plantations:
            if col == 'bg_logging':
                to_add.append(df['capped_resources'].map(lambda x: x.get(col, 0)))
            else:
                to_add.append(df['arable_resources'].map(lambda x: col in x) * df['arable_land'])
        return pd.concat(to_add, axis='columns', keys=plantations)

    def one_hot_fish(extra=None):
        basic_resources = ['bg_fishing', 'bg_whaling']
        to_add = []
        if extra is not None:
            basic_resources = list(map(lambda x: x+f'_{extra}', basic_resources))
        for col in basic_resources:
            if extra is None:
                to_add.append(df['capped_resources'].map(lambda x: x.get(col, 0)))
            elif extra == 'log':
                to_add.append(safe_log(df['capped_resources'].map(lambda x: x.get(col, 0))))
            elif extra == 'square':
                to_add.append((df['capped_resources'].map(lambda x: x.get(col, 0)))**2)
            # to_add.append(df['capped_resources'].map(lambda x: x.get(col, 0)))
        return pd.concat(to_add, axis='columns', keys=basic_resources)

    def one_hot_regions(extra=None):
        to_add = []
        regions = list(filter(lambda x: x.find('sea') <0 ,os.listdir(resource_data_dir)))
        if extra is not None:
            regions = list(map(lambda x: x+f'_{extra}', regions))
        for col in regions:
            if extra is None:
                to_add.append(df['region'].map(lambda x: col==x)*df['arable_land'])
            elif extra == 'log':
                to_add.append(safe_log(df['region'].map(lambda x: col==x)*df['arable_land']))
            elif extra == 'square':
                to_add.append((df['region'].map(lambda x: col==x)*df['arable_land'])**2)
            # to_add.append(df['region'].map(lambda x: x== col) * df['arable_land'])
        return pd.concat(to_add, axis='columns',
                         keys=regions)

    def one_hot_state_traits():
        # traits = ['malaria', 'severe_malaria', 'soil', 'harbor',
        #        'whaling', 'fishing', 'river', 'delta', 'desert', 'mountain', 'valley', 'forest', 'lake']
        traits = ['river']
        to_add = []
        for col in traits:
            to_add.append(df['state_traits'].map(lambda x: x.get(col, False)))
        return pd.concat(to_add, axis='columns', keys=traits)

    out = []
    for name in sets:
        if 'log' in name:
            name = name.replace('_log', '')
            extra = 'log'
            out.append(locals()[f'one_hot_{name}'](extra))
        elif 'square' in name:
            name = name.replace('_square', '')
            extra = 'square'
            out.append(locals()[f'one_hot_{name}'](extra))
        else:
            out.append(locals()[f'one_hot_{name}']())
    return out

def prepare_df(sets_to_load, sets_to_regress, old_world_only=False):
    sets_to_load += ['pops']
    states_data = combine_state_data(list(map(lambda x: globals()[f'get_state_{x}'](), sets_to_load)))
    df = pd.DataFrame.from_dict(states_data, orient='index')
    if old_world_only:
        df = df[df['region'].map(lambda x: 'america' in x or 'austra' in x) == False]
    one_hots = get_one_hots(sets_to_regress, df)
    # print(sets_to_regress)
    to_drop = ['subsistence_building', 'arable_resources', 'capped_resources', 'state_traits', 'impassables',
               'region'] * ('resources' in sets_to_load)
    sets_to_load.remove('resources')
    sets_to_load.remove('pops')
    to_drop += sets_to_load
    df = df.drop(to_drop,
                 axis=1)
    df = pd.concat([df] + one_hots, axis='columns')
    df = df.drop('arable_land', axis=1)
    return df

def test_predictor(sets_to_load, sets_to_regress, thresholds=[0.5], old_world_only=False):
    from sklearn.linear_model import LinearRegression
    df = prepare_df(sets_to_load, sets_to_regress, old_world_only)
    print('num_states:', len(df.index))
    wrongs = {threshold: set([]) for threshold in thresholds}
    too_highs = {threshold: set([]) for threshold in thresholds}
    too_lows = {threshold: set([]) for threshold in thresholds}
    rights = {threshold: set([]) for threshold in thresholds}

    for i in range(len(df.index)):
        train = pd.concat([df.iloc[:i], df.iloc[i + 1:]])
        test = df.iloc[[i]]
        y_train = train['pop']
        x_train = train.drop('pop', axis=1)
        y_test = test['pop']
        x_test = test.drop('pop', axis=1)

        model = LinearRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        for threshold in thresholds:
            y_too_high = y_pred * (1 - threshold) > y_test
            y_too_low = y_pred * (1 + threshold) < y_test
            show = pd.DataFrame(y_test)
            show['pred'] = y_pred
            show['too_high'] = y_too_high
            show['too_low'] = y_too_low
            too_highs[threshold] = too_highs[threshold].union(set(show[show['too_high'] == True].index))
            too_lows[threshold] = too_lows[threshold].union(set(show[show['too_low'] == True].index))
            wrongs[threshold] = too_highs[threshold].union(too_lows[threshold])
            rights[threshold] = set(df.index) - wrongs[threshold]
    print('num_wrongs:', {threshold: len(wrongs[threshold]) for threshold in thresholds})
    return rights, wrongs, too_highs, too_lows

def get_predictor(sets_to_load, sets_to_regress, old_world_only=False):
    from sklearn.linear_model import LinearRegression
    df = prepare_df(sets_to_load, sets_to_regress, old_world_only)
    df.drop('pop', axis=1).sample(20).to_csv('sample_df.csv')
    model = LinearRegression(positive=True)
    model.fit(df.drop('pop', axis=1), df['pop'])
    coefs = dict(zip(list(model.feature_names_in_) + ['intercept'] , list(model.coef_) + [model.intercept_]))
    print(coefs)
    suffix = '_'.join(sets_to_regress)
    json.dump(coefs, open(f'baseline_pop_equation_{suffix}.json', 'w'))
    return model

if __name__ == '__main__':
    get_predictor(['resources'], ['regions'])
