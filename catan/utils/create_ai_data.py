import json
import random

import pandas as pd
import numpy as np

from .compiled_parameters import *


def build_ai_data(ii, winner, static, states, actions, options):
    x1 = []
    x2 = []
    y = []
    slices = generate_slices(options)
    for player in ['white', 'blue', 'orange']:
        state_data = list \
            (map(
                lambda z: static + list(map(lambda a: a if a not in seating.keys() else seating[player][a], z['board'])) \
                          + z['self'] + z['left'] + z['right'], map(lambda x: {'board': x[3:][:-60],
                                                                               seating[player]['white']: x[3:][-60:-40],
                                                                               seating[player]['blue']: x[3:][-40:-20],
                                                                               seating[player]['orange']: x[3:][-20:]},
                                                                    states[slices['build'][player]])))
        action_data = actions[slices['build'][player]]
        option_data = options[slices['build'][player]]
        if player == winner:
            for i in range(len(state_data)):
                choices = list(map(lambda x: x[0], filter(lambda z: z[1] == 1, option_data[i])))
                if len(action_data[i]) > 0:
                    x1.append(state_data[i])
                    x2.append([action_data[i][0][1]])
                    y.append([1])
                    x1.append(state_data[i])
                    x2.append(['none'])
                    y.append([0])
                    choices.remove(action_data[i][0][1])
                else:
                    x1.append(state_data[i])
                    x2.append(['none'])
                    y.append([1])
                while len(choices) > 1:
                    x1.append(state_data[i])
                    chosen = random.choice(choices)
                    x2.append([chosen])
                    y.append([0])
                    choices.remove(chosen)
    print(len(y))
    pd.concat([pd.DataFrame(x1), pd.DataFrame(x2)], axis=1).to_csv(f'intermediate_data/build/x/build_x_{ii}.csv',
                                                                   header=False, index=False)
    pd.DataFrame(y).to_csv(f'intermediate_data/build/y/build_y_{ii}.csv', header=False, index=False)


def trade_ai_data(ii, winner, static, states, actions, options):
    x1 = []
    x2 = []
    y = []
    slices = generate_slices(options)
    for player in ['white', 'blue', 'orange']:
        state_data = list(map(lambda z: static + list(
            map(lambda a: a if a not in seating.keys() else seating[player][a], z['board'])) \
                                        + z['self'] + z['left'] + z['right'], map(lambda x: {'board': x[3:][:-60],
                                                                                             seating[player][
                                                                                                 'white']: x[3:][
                                                                                                           -60:-40],
                                                                                             seating[player][
                                                                                                 'blue']: x[3:][
                                                                                                          -40:-20],
                                                                                             seating[player][
                                                                                                 'orange']: x[3:][
                                                                                                            -20:]},
                                                                                  states[slices['trade'][player]])))
        action_data = actions[slices['trade'][player]]
        option_data = options[slices['trade'][player]]
        if player == winner:
            for i in range(len(state_data)):
                choices = option_data[i].copy()
                if len(action_data[i]) > 0:
                    x1.append(state_data[i])
                    x2.append(action_data[i][0][1:3])
                    y.append([1])
                    x1.append(state_data[i])
                    x2.append(['none', 'none'])
                    y.append([0])
                    choices.remove(action_data[i][0][1:3])
                else:
                    x1.append(state_data[i])
                    x2.append(['none', 'none'])
                    y.append([1])
                j = 1
                while len(choices) > 0 and j < 3:
                    x1.append(state_data[i])
                    chosen = random.choice(choices)
                    x2.append(chosen)
                    y.append([0])
                    choices.remove(chosen)
    print(len(y))
    pd.concat([pd.DataFrame(x1), pd.DataFrame(x2)], axis=1).to_csv(f'intermediate_data/trade/x/trade_x_{ii}.csv',
                                                                   header=False, index=False)
    pd.DataFrame(y).to_csv(f'intermediate_data/trade/y/trade_y_{ii}.csv', header=False, index=False)


def road_ai_data(ii, winner, static, states, actions, options):
    x1 = []
    x2 = []
    y = []
    slices = generate_slices(options)
    for player in ['white', 'blue', 'orange']:
        state_data = list(map(lambda z: static + list(
            map(lambda a: a if a not in seating.keys() else seating[player][a], z['board'])) \
                                        + z['self'] + z['left'] + z['right'], map(lambda x: {'board': x[3:][:-60],
                                                                                             seating[player][
                                                                                                 'white']: x[3:][
                                                                                                           -60:-40],
                                                                                             seating[player][
                                                                                                 'blue']: x[3:][
                                                                                                          -40:-20],
                                                                                             seating[player][
                                                                                                 'orange']: x[3:][
                                                                                                            -20:]},
                                                                                  states[slices['build'][player]])))
        # print(state_data)
        action_data = actions[slices['build'][player]]
        option_data = options[slices['build'][player]]
        if player == winner:
            for i in range(len(state_data)):
                if len(action_data[i]) > 0 and action_data[i][0][1] == 'road':
                    state_data[i][-58] -= 1
                    state_data[i][-57] -= 1
                    choices = list(map(lambda x: edge_dict[tuple(x)], option_data[i][0][2].copy()))
                    correct = edge_dict[tuple(json.loads(action_data[i][1][1].replace('(', '[').replace(')', ']')))]
                    x1.append(state_data[i])
                    x2.append([correct])
                    y.append([1])
                    choices = [choice for choice in choices if choice != correct]
                    j = 1
                    while len(choices) > 0 and j < 5:
                        x1.append(state_data[i])
                        remove = random.choice(choices)
                        x2.append([remove])
                        y.append([0])
                        choices.remove(remove)
                        j += 1
    print(len(y))
    if len(x2) > 0:
        # placement = pd.DataFrame(x2)
        # edges = []
        # for i in range(72):
        #     edges.append(placement[0].map(lambda x: int(x==i)))
        edges = [pd.DataFrame(x2)]
        pd.concat([pd.DataFrame(x1)] + edges, axis=1).to_csv(f'intermediate_data/road/x/road_x_{ii}.csv',
                                                                       header=False, index=False)
        pd.DataFrame(y).to_csv(f'intermediate_data/road/y/road_y_{ii}.csv', header=False, index=False)

