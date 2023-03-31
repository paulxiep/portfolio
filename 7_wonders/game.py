import json
import random
from functools import reduce
from elements import Card, Wonder
from effects import *
from gymnasium import Env
from gymnasium.spaces import MultiBinary, Box, Tuple, Discrete, MultiDiscrete
from gymnasium.spaces.utils import flatten_space
import numpy as np

class Game(Env):
    def __init__(self, players):
        super().__init__()
        self.players = players
        self.cards = {1: [], 2: [], 3: []}
        # choose from among 80 cards, pay left and/or right, choose what to do with card
        self.action_space = Tuple((Discrete(80), Box(np.zeros(2), np.ones(2)), Discrete(3)))
        board_space = Tuple((Box(np.zeros(15), np.ones(15)), # production
                            Box(np.zeros(15), np.ones(15)),  # sellable
                            Box(np.zeros(4), np.ones(4)), # coins army points wonder_stage
                            Box(np.zeros(7), np.ones(7)),  # colors
                            Box(np.zeros(4), np.ones(4)),  # science
                            MultiBinary(24), # chain
                            MultiBinary(3), # discount
                            MultiBinary(10), # guilds
                            MultiBinary(42) # wonder_choice
                            ))
        self.observation_space = Tuple((MultiBinary(80), board_space, board_space, board_space,
                                  Discrete(4) # choose, play for price, choose and place for free, idle
                                    ))

    def print_action(self):
        print(flatten_space(self.action_space).shape)

    def print_observation(self):
        print(flatten_space(self.observation_space).shape)

    def step(self, action_n):
        '''
        still dummy
        '''
        obs_n = list()
        reward_n = list()
        done_n = list()
        info_n = {'n': []}
        # ...
        return obs_n, reward_n, done_n, info_n

    @staticmethod
    def translate_requirements(requirements):
        if requirements is None:
            return {}
        elif requirements.get('resources', False):
            out = {}
            for r in requirements['resources'].replace('C', 'B'):
                out[r] = out.get(r, 0) + 1
            return out
        else:
            return {'C': requirements['gold']}

    def get_wonders(self):
        with open('v2_wonders.json', 'r') as f:
            wonders = json.load(f)
        self.wonders = [Wonder.from_dict(wonders.pop(random.randrange(len(wonders)))) \
                        for _ in range(self.players)]

    def build_deck(self):
        def upper_first(string):
            return string[0].upper() + string[1:]
        with open('v2_cards.json', 'r') as f:
            cards = json.load(f)
        for i in range(1, 4):
            self.cards[i] = reduce(list.__add__, [[Card(card['name'], card['color'],
                                                        Effect.from_dict(card['effect']),
                                                        # globals()[upper_first(list(card['effect'].keys())[0])].from_dict(card['effect']),
                                  self.translate_requirements(card.get('requirements', None)),
                                  card.get('chainParent', None),
                                  card.get('chainChildren', []))] \
                                 * card['countPerNbPlayer'][str(self.players)] \
                                 for card in cards[f'age{i}']['cards']]) + (i==3) * \
                                    random.sample([Card(card['name'], card['color'],
                                                        Effect.from_dict(card['effect']),
                                                        # globals()[upper_first(list(card['effect'].keys())[0])].from_dict(card['effect']),
                                  self.translate_requirements(card.get('requirements', None)),
                                  card.get('chainParent', None),
                                  card.get('chainChildren', [])) for card in cards['guildCards']], self.players+2)