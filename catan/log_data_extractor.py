import json
import re
from functools import reduce
import random
import pandas as pd
from catan_units.compiled_parameters import *
from multiprocessing import Pool
import os

seating = {'white': {'blue': 'left', 'orange': 'right', 'white': 'self'},
          'blue': {'orange': 'left', 'white': 'right', 'blue': 'self'},
          'orange': {'white': 'left', 'blue': 'right', 'orange': 'self'}}

def extract_states(state_list):
    '''
    1. state
    2. corners
    3. edges
    4. hexes
    5-7. players
    '''

    def extract_state(state_set):
        #         print(state_set, '\n')
        try:
            return json.loads(state_set[0].replace('\'', '"').replace('(', '[').replace(')', ']')) + \
                   reduce(list.__add__, json.loads(state_set[1].replace('None', '"none"').replace('\'', '"'))) + \
                   json.loads(state_set[2].replace('None', '"none"').replace('\'', '"')) + \
                   json.loads(state_set[3].replace('False', '0').replace('True', '1')) + \
                   json.loads(state_set[4]) + \
                   json.loads(state_set[5]) + \
                   json.loads(state_set[6])
        except:
            return state_set

    i = 0
    states = []
    while 7 * i + 3 < len(state_list):
        states.append(extract_state(state_list[7 * i + 3: 7 * i + 10]))
        i += 1
    return states[:-1]


def extract_actions(action_list):
    def extract_action(action):
        #         if 'setup1' in action:
        if 'buy' in action:
            return 'buy', action.split(' ')[-1]
        elif 'setup_1' in action or 'setup_2' in action:
            return action
        elif 'used' in action:
            return 'used', action.split(' ')[-1]
        elif 'plenty' in action:
            return 'plenty', action.split(' ')[-3]
        elif 'takes' in action:
            return 'monopoly', action.split(' ')[-3]
        elif 'robber' in action:
            return 'robber', re.findall('\(\d*, \d*\)', action)[0]
        elif 'settlement' in action:
            return 'settlement', re.findall('\(\d*, \d*\)', action)[0]
        elif 'city' in action:
            return 'city', re.findall('\(\d*, \d*\)', action)[0]
        elif 'road' in action:
            return 'road', re.findall('\(\d*, \d*\)', action)[0]
        elif 'development' in action:
            pass
        elif 'exchanged' in action:
            return 'trade', action.split(' ')[-4], action.split(' ')[-1]
        elif 'rolled' in action:
            pass
        elif 'steals' in action:
            pass
        elif 'collects' in action:
            pass
        elif 'discards' in action:
            pass
        elif len(action) == 0:
            pass
        else:
            return action

    return list(filter(lambda x: x is not None, map(extract_action, action_list)))
def extract_options(option_list):
    def extract_option(option):
        return json.loads(option.replace('(', '[').replace(')', ']').replace('\'', '"').replace('False', '0').replace('True', '1'))
    length = len(option_list)//2
    return [list(map(extract_option, option_list))[2*i+1] for i in range(length)]

def extract_all(action_file):
    def build_ai_data(winner, static, states, actions, options):
        x1 = []
        x2 = []
        y = []
        for player in ['white', 'blue', 'orange']:
            state_data = list(map(lambda z: static + list(map(lambda a: a if a not in seating.keys() else seating[player][a], z['board'])) \
                                  + z['self'] + z['left'] + z['right'], map(lambda x: {'board': x[3:][:-60],
                                             seating[player]['white']: x[3:][-60:-40],
                                             seating[player]['blue']: x[3:][-40:-20],
                                             seating[player]['orange']: x[3:][-20:]}, states[slices['build'][player]])))
            action_data = actions[slices['build'][player]]
            option_data = options[slices['build'][player]]
            if player == winner:
                for i in range(len(state_data)):
                    choices = list(map(lambda x: x[0], filter(lambda z: z[1]==1, option_data[i])))
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
    def trade_ai_data(winner, static, states, actions, options):
        x1 = []
        x2 = []
        y = []
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
    def road_ai_data(winner, static, states, actions, options):
        x1 = []
        x2 = []
        y = []
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
            action_data = actions[slices['build'][player]]
            option_data = options[slices['build'][player]]
            if player == winner:
                for i in range(len(state_data)):
                    if len(action_data[i]) > 0 and action_data[i][0][1] == 'road':
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
        pd.concat([pd.DataFrame(x1), pd.DataFrame(x2)], axis=1).to_csv(f'intermediate_data/road/x/road_x_{ii}.csv',
                                                                       header=False, index=False)
        pd.DataFrame(y).to_csv(f'intermediate_data/road/y/road_y_{ii}.csv', header=False, index=False)
    ii = int(action_file.split('_')[1].split('.')[0])
    with open(f'action_logs/action_{ii}.txt', 'r') as f:
        action_log = f.read()
    with open(f'state_logs/state_{ii}.txt', 'r') as f:
        state_log = f.read()
    with open(f'option_logs/option_{ii}.txt', 'r') as f:
        option_log = f.read()
    harbors = json.loads(state_log[7:].split('\nstate: ')[0].replace('\'', '"'))
    resources = json.loads(state_log[7:].split('\nstate: ')[1].replace('\'', '"'))
    numbers = json.loads(state_log[7:].split('\nstate: ')[2])
    winner = state_log.split(' ')[-1][:-1]
    try:
        assert winner in ['white', 'blue', 'orange']
    except:
        return None

    static = harbors + resources + numbers

    states = extract_states(state_log[7:].split('\nstate: '))
    try:
        assert len(states) < 1215
    except:
        return None
    actions = extract_actions(action_log[8:].split('\naction: '))
    options = extract_options(option_log[8:].split('\noption: '))
    actions.pop(0)
    all_actions = []
    while len(actions) > 0:
        turn_actions = []
        #     print(action)
        action = actions.pop(0)
        while isinstance(action, tuple):
            turn_actions.append(list(action))
            action = actions.pop(0)
        all_actions.append(turn_actions)

    slices = {'first_settlement': {
                'white': 0,
                'blue': 2,
                'orange': 4
            },
             'second_settlement': {
                'white': 12,
                'blue': 9,
                'orange': 6
             },
             'initial_road': {
                'white': slice(1, 15, 13),
                'blue': slice(3, 12, 8),
                'orange': slice(5, 9, 3)
             },
             'knight': {
                'white': slice(15, len(options), 12),
                'blue': slice(19, len(options), 12),
                'orange': slice(23, len(options), 12)
            },
             'development': {
                'white': slice(16, len(options), 12),
                'blue': slice(20, len(options), 12),
                'orange': slice(24, len(options), 12)
            },
             'trade': {
                'white': slice(17, len(options), 12),
                'blue': slice(21, len(options), 12),
                'orange': slice(25, len(options), 12)
            },
             'build':{
                'white': slice(18, len(options), 12),
                'blue': slice(22, len(options), 12),
                'orange': slice(26, len(options), 12)
            }}

    road_ai_data(winner, static, states, all_actions, options)
    # trade_ai_data(winner, static, states, all_actions, options)
    # build_ai_data(winner, static, states, all_actions, options)

if __name__ == '__main__':
    action_files = os.listdir('action_logs')[:5]
    # extract_all(action_files[0])
    with Pool(os.cpu_count()*3//4) as p:
        p.map(extract_all, action_files)
    # extract_all(1)