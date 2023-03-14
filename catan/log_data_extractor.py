import os
import re
from functools import reduce
from multiprocessing import Pool

from utils.create_ai_data import *


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
        def transform_corner(settlement, city):
            if settlement != 'none':
                out1 = settlement
                out2 = 1
            elif city != 'none':
                out1 = city
                out2 = 2
            else:
                out1 = 'none'
                out2 = 0
            return out1, out2

        try:
            corners = reduce(list.__add__, json.loads(state_set[1].replace('None', '"none"').replace('\'', '"')))
            # print(corners)
            for i in range(len(corners)//2):
                temp1, temp2 = transform_corner(corners[2*i], corners[2*i+1])
                corners[2*i] = temp1
                corners[2*i+1] = temp2
            # print(corners)
            return json.loads(state_set[0].replace('\'', '"').replace('(', '[').replace(')', ']')) + \
                   corners + \
                   json.loads(state_set[2].replace('None', '"none"').replace('\'', '"')) + \
                   [json.loads(state_set[3].replace('False', '0').replace('True', '1')).index(1)] + \
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
        return json.loads(
            option.replace('(', '[').replace(')', ']').replace('\'', '"').replace('False', '0').replace('True', '1'))

    length = len(option_list) // 2
    return [list(map(extract_option, option_list))[2 * i + 1] for i in range(length)]


def extract_all(action_file):
    def pips(number):
        return (6 - abs(7 - number)) * (number != 0)
    ii = int(action_file.split('_')[1].split('.')[0])
    with open(f'ai_logs/road_ai_logs/action_logs/action_{ii}.txt', 'r') as f:
        action_log = f.read()
    with open(f'ai_logs/road_ai_logs/state_logs/state_{ii}.txt', 'r') as f:
        state_log = f.read()
    with open(f'ai_logs/road_ai_logs/option_logs/option_{ii}.txt', 'r') as f:
        option_log = f.read()
    harbors = json.loads(state_log[7:].split('\nstate: ')[0].replace('\'', '"'))
    resources = json.loads(state_log[7:].split('\nstate: ')[1].replace('\'', '"'))
    numbers = list(map(pips, json.loads(state_log[7:].split('\nstate: ')[2])))
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

    road_ai_data(ii, winner, static, states, all_actions, options)
    # trade_ai_data(ii, winner, static, states, all_actions, options)
    # build_ai_data(ii, winner, static, states, all_actions, options)


if __name__ == '__main__':
    action_files = os.listdir('ai_logs/road_ai_logs/action_logs')
    action_files = list(filter(lambda x: int(x.split('.')[0].split('_')[1])<=96000, action_files))
    # extract_all(action_files[0])
    with Pool(os.cpu_count() * 1 // 4) as p:
        p.map(extract_all, action_files)
    # extract_all(1)
