import os
import cv2
import numpy as np
from functools import reduce, partial
from multiprocessing import Pool
from ml import test_predictor
from data_preparation import get_states

data_dir = '/Games/Steam/steamapps/common/Victoria 3/game'
map_data_file = os.path.join(data_dir, 'map_data/provinces.png')

def get_states_reverse():
    def accu_state(substates):
        return reduce(list.__add__, substates.values())
    states_reverse = {}
    for k, v in {k: accu_state(v) for k, v in get_states().items()}.items():
        for item in v:
            states_reverse[item[0]+item[1:].upper()] = k
    return states_reverse

def get_raw_map():
    return cv2.imread(map_data_file)

def get_prediction_map(rights, wrongs, too_highs, too_lows, threshold, states_reverse, dir):
    wrongs = {item: True for item in wrongs}
    rights = {item: True for item in rights}
    too_highs = {item: True for item in too_highs}
    too_lows = {item: True for item in too_lows}
    from matplotlib.colors import to_rgb
    def right_or_wrong(x):
        color = 'x' + hex(x[2])[2:].upper().zfill(2) + hex(x[1])[2:].upper().zfill(2) + hex(x[0])[2:].upper().zfill(2)
        if too_highs.get(states_reverse.get(color, ''), False):
            return to_rgb('#000000')
        elif too_lows.get(states_reverse.get(color, ''), False):
            return to_rgb('#FFFFFF')
        elif rights.get(states_reverse.get(color, ''), False):
            return to_rgb('#888888')
        else:
            return to_rgb('#' + color[1:])

    color_map = np.apply_along_axis(right_or_wrong, 1, np.reshape(get_raw_map(), [-1, 3]))
    color_map = np.reshape(color_map, [3616, 8192, 3])
    color_map = np.stack([color_map[:, :, 2], color_map[:, :, 1], color_map[:, :, 0]], axis=2)
    cv2.imwrite(f'{dir}/{threshold:3.2f}.jpg', (color_map*255).astype(np.uint8))
    return color_map

def get_prediction_maps(sets_to_load, sets_to_regress, thresholds=[0.5], old_world_only=False):
    rights, wrongs, too_highs, too_lows = test_predictor(sets_to_load, sets_to_regress, thresholds, old_world_only)
    states_reverse = get_states_reverse()
    inputs = [(rights[threshold], wrongs[threshold],
               too_highs[threshold], too_lows[threshold], threshold) for threshold in thresholds]
    world_type = 'old_'*old_world_only + 'world'
    dir = f'highs_and_lows/{"_".join(sets_to_regress)}/{world_type}'
    if not os.path.exists(dir):
        os.makedirs(dir)
    with Pool(12) as p:
        p.starmap(partial(get_prediction_map, states_reverse=states_reverse, dir=dir), inputs)

if __name__ == '__main__':
    sets_to_load = ['resources', 'buildings', 'terrain']
    sets_to_regress = ['subsistence', 'fish', 'regions']
    old_world_only = False
    get_prediction_maps(sets_to_load, sets_to_regress, np.arange(0.05, 0.51, 0.05), old_world_only)

