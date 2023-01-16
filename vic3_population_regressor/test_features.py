import os
import json
import pandas as pd
import numpy as np
from functools import reduce
from itertools import combinations
from multiprocessing import Pool
from ml import test_predictor

# using the full list will take ages to run
# all_possible = ['subsistence', 'farms', 'fish', 'non_food', 'state_traits', 'buildings', 'terrain', 'coast', 'regions']
all_possible = ['subsistence', 'farms', 'fish', 'regions']
# none of the log or squared features proved useful, but feel free to try
# all_possible = all_possible + list(map(lambda x: x + '_log', all_possible))
# all_possible = all_possible + list(map(lambda x: x + '_log', all_possible)) + \
               # list(map(lambda x: x + '_square', all_possible))

def test_features(combination):
    result_df = pd.DataFrame(columns=all_possible + ['threshold', 'all', 'rights', 'wrongs', 'too_highs', 'too_lows'])
    i = 1
    thresholds = np.arange(0.05, 0.51, 0.05)
    for old_world_only in [False]:
        rights, wrongs, too_highs, too_lows = test_predictor(['resources', 'buildings', 'terrain'],
                                                    list(combination),
                                                        thresholds=thresholds, old_world_only=old_world_only)
        for threshold in thresholds:
            result_df.loc[i] = [(possible in combination) for possible in all_possible] + \
                               [threshold, len(rights[threshold]) + len(wrongs[threshold]),
                                len(rights[threshold]), len(wrongs[threshold]), len(too_highs[threshold]),
                                len(too_lows[threshold])]
            i += 1
    return result_df

if __name__ == '__main__':
    # this commented out line is for running single-thread for debug
    # results = list(map(test_features, reduce(list.__add__, [list(combinations(all_possible, n+1)) for n in range(len(all_possible))])))
    with Pool(12) as p:
        results = p.map(test_features, reduce(list.__add__, [list(combinations(all_possible, n+1)) for n in range(len(all_possible))]))
    pd.concat(results, axis='index', ignore_index=True).to_csv('world_metrics_fe.csv')