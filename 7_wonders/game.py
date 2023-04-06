import json
import random
from functools import reduce

import pandas as pd
import numpy as np
from gymnasium import Env
from gymnasium.spaces import MultiBinary, Box, Tuple, Discrete
from gymnasium.spaces.utils import flatten_space

from elements import Card, Wonder, Player, Board, DQNPlayer
from src.constants import *
from src.utils import *


class Game(Env):
    def __init__(self, id, players):
        super().__init__()
        self.id = id
        self.n_players = players
        self.cards = {1: [], 2: [], 3: []}
        # choose from among 80 cards, pay left and/or right, choose what to do with card
        self.action_space = {'choose': Discrete(80),
                             'play': Discrete(3),
                             'pay': Box(np.zeros(2), np.ones(2))}
        board_space = Tuple((Box(np.zeros(15), np.ones(15)),  # production
                             Box(np.zeros(15), np.ones(15)),  # sellable
                             Box(np.zeros(4), np.ones(4)),  # coins army points wonder_stage
                             Box(np.zeros(7), np.ones(7)),  # colors
                             Box(np.zeros(4), np.ones(4)),  # science
                             MultiBinary(24),  # chain
                             MultiBinary(3),  # discount
                             MultiBinary(15),  # guilds
                             MultiBinary(5),  # wonder_effects
                             MultiBinary(42)  # wonder_choice
                             ))
        self.observation_space = flatten_space(Tuple((MultiBinary(80), board_space, board_space, board_space,
                                        Box(np.zeros(1), np.ones(1)),  # game progress; order of card in play
                                        Discrete(3),  # choose, play for price, pay, play_discarded/idle
                                        Discrete(1)  # reverse passing direction
                                        )))

    def setup(self):
        self.build_deck()
        self.get_wonders()
        self.create_players()
        self.nth = 0
        self.action = 0
        self.memory = []
        self.discarded = []
        self.era = 1
        for i in range(self.n_players):
            self.players[i].hand = self.cards[self.era][(i*7): ((i+1)*7)]
            self.players[i].cards = self.cards[self.era][(i * 7): ((i + 1) * 7)]

    def run_and_save(self):
        '''
        not meant to be used, for testing memory data
        '''
        self.collect(True)
        pd.DataFrame(reduce(list.__add__, [squash_idle(list(pd.DataFrame(self.memory)[i])) for i in range(self.n_players)])).to_csv(f'games/game_{self.id}.csv', header=False, index=None)

    def end(self):
        obs_n = ['idle'] * self.n_players
        done_n = [1] * self.n_players
        info_n = None
        reward_n = []
        for i in range(self.n_players):
            self.players[i].calculate_guilds()
            reward_n.append(
                self.players[i].calculate_science() +
                self.players[i].board.coins // 3 +
                self.players[i].board.points
            )
        return obs_n, reward_n, done_n, info_n, 21, 0

    def collect(self, training):
        '''
        for simulating whole game
        '''
        done = False
        obs_n = list(map(lambda i: self.players[i].prepare_obs(), range(self.n_players)))
        while not done:
            action_n = list(map(lambda i: self.players[i].select_action(obs_n[i], training), range(self.n_players)))
            next_obs_n, reward_n, done_n, info_n, nth, action = self.step(action_n)
            self.memory.append([(obs_n[i], action_n[i], next_obs_n[i], reward_n[i], done_n[i], nth, action) for i in range(self.n_players)])
            obs_n = next_obs_n
            done = bool(done_n[0])

    def turn(self, training):
        '''
        not used yet, for processing single turn
        '''
        obs_n = list(map(lambda i: self.players[i].prepare_obs(), range(self.n_players)))
        action_n = list(map(lambda i: self.players[i].select_action(obs_n[i], training), range(self.n_players)))
        next_obs_n, reward_n, done_n, info_n, nth, action = self.step(action_n)
        self.memory.append([(obs_n[i], action_n[i], next_obs_n[i], reward_n[i], done_n[i], nth, action) for i in range(self.n_players)])

    def step(self, action_n):
        nth, naction = self.nth, self.action
        if False:#self.nth == 20 and self.action == 3:
            pass#return self.end()
        elif self.action == 0:
            self.action = 1
            obs_n = []
            for i in range(self.n_players):
                if self.nth not in [6, 13, 20] or self.players[i].board.wonder_effects['PLAY_LAST_CARD']:
                    chosen = {v: k for k, v in card_dict.items()}[action_n[i]]
                    for cid in range(len(self.players[i].hand)):
                        if self.players[i].hand[cid].name == chosen:
                            self.players[i].chosen =  self.players[i].hand.pop(cid)
                            break
                    if (self.nth in [0, 7, 14] and self.players[i].board.wonder_effects['FIRST_FREE_PER_AGE']) or \
                            (self.nth in [5, 12, 19] and self.players[i].board.wonder_effects['LAST_FREE_PER_AGE']) or \
                            (self.players[i].board.wonder_effects['FIRST_FREE_PER_COLOR'] and self.players[i].board.colors[self.players[i].chosen.color.lower()]==0):
                        self.players[i].apply_card(self.players[i].chosen)
                    obs_n.append(self.players[i].prepare_obs())
                else:
                    obs_n.append('idle')
        elif self.action == 1:
            self.action = 2
            obs_n = []
            for i in range(self.n_players):
                if self.nth not in [6, 13, 20] or self.players[i].board.wonder_effects['PLAY_LAST_CARD']:
                    if (self.nth in [0, 7, 14] and self.players[i].board.wonder_effects['FIRST_FREE_PER_AGE']) or \
                            (self.nth in [5, 12, 19] and self.players[i].board.wonder_effects['LAST_FREE_PER_AGE']) or \
                            (self.players[i].board.wonder_effects['FIRST_FREE_PER_COLOR'] and self.players[i].board.colors[self.players[i].chosen.color.lower()]==0):
                        obs_n.append('idle')
                    else:
                        self.players[i].action = action_n[i]
                        if action_n[i] == 2:
                            self.players[i].board.coins += 3
                            self.discarded.append(self.players[i].chosen)
                            obs_n.append('idle')
                        else:
                            obs_n.append(self.players[i].prepare_obs())
                else:
                    obs_n.append('idle')
        elif self.action == 2:
            self.action = 3
            obs_n = []
            for i in range(self.n_players):
                if self.nth not in [6, 13, 20] or self.players[i].board.wonder_effects['PLAY_LAST_CARD']:
                    if ((self.nth in [0, 7, 14] and self.players[i].board.wonder_effects['FIRST_FREE_PER_AGE']) or \
                            (self.nth in [5, 12, 19] and self.players[i].board.wonder_effects['LAST_FREE_PER_AGE']) or \
                            (self.players[i].board.wonder_effects['FIRST_FREE_PER_COLOR'] and self.players[i].board.colors[self.players[i].chosen.color.lower()]==0)):
                        obs_n.append('idle')
                    else:
                        if self.players[i].action != 2:
                            if action_n[i][0] == 0 and action_n[i][1]==0:
                                self.players[i].board.coins -= int(round(action_n[i][2]*20))
                            else:
                                self.players[i].board.coins -= int(round((action_n[i][0] + action_n[i][1])*20))
                                self.players[i].left.coins += int(round(action_n[i][0]*20))
                                self.players[i].right.coins += int(round(action_n[i][1]*20))
                            if self.players[i].action == 0:
                                self.players[i].apply_card(self.players[i].chosen)
                            elif self.players[i].action == 1:
                                self.players[i].build_wonder()
                        if not self.players[i].board.wonder_effects['PLAY_DISCARDED']:
                            obs_n.append('idle')
                        else:
                            if len(self.discarded) > 0:
                                self.players[i].cards = self.discarded
                                obs_n.append(self.players[i].prepare_obs())
                            else:
                                obs_n.append('idle')
                else:
                    obs_n.append('idle')
            if self.nth in [6, 13, 20]:  # discard at end of era
                for i in range(self.n_players):
                    for card in self.players[i].hand:
                        self.discarded.append(card)
        elif self.action == 3:
            obs_n = []
            for i in range(self.n_players):
                if self.nth not in [6, 13, 20] or self.players[i].board.wonder_effects['PLAY_LAST_CARD']:
                    if self.players[i].board.wonder_effects['PLAY_DISCARDED']:
                        self.players[i].board.wonder_effects['PLAY_DISCARDED'] = False
                        if len(self.discarded) > 0:
                            chosen = {v: k for k, v in card_dict.items()}[action_n[i]]
                            for cid in range(len(self.players[i].hand)):
                                if self.players[i].cards[cid].name == chosen:
                                    self.players[i].chosen = self.players[i].cards.pop(cid)
                            self.players[i].apply_card(self.players[i].chosen)
            if self.nth not in [6, 13, 20]: # not end of era
                if self.nth not in range(7, 14):  #pass clockwise
                    buffer_from = self.players[-1].hand
                    for i in range(self.n_players):
                        buffer_to = self.players[i].hand
                        self.players[i].hand = buffer_from
                        self.players[i].cards = self.players[i].hand
                        buffer_from = buffer_to
                else:  #pass counter clockwise
                    buffer = self.players[0].hand
                    for i in range(self.n_players):
                        if i < self.n_players - 1:
                            self.players[i].hand = self.players[i+1].hand
                        else:
                            self.players[i].hand = buffer
                        self.players[i].cards = self.players[i].hand
            else: # end of era
                for opponent in [self.players[i-1], self.players[i-6]]:
                    if self.players[i].board.army < opponent.board.army:
                        self.players[i].board.points -= 1
                    elif self.players[i].board.army > opponent.board.army:
                        self.players[i].board.points += 1 + (self.era-1) * 2

                if self.nth == 20: # end of game
                    pass
                else:   # start next era
                    if self.nth == 13: # start 3rd era
                        self.era = 3
                    else:
                        self.era = 2
                    for i in range(self.n_players):
                        self.players[i].hand = self.cards[self.era][(i*7): ((i+1)*7)]
                        self.players[i].cards = self.cards[self.era][(i * 7): ((i + 1) * 7)]

            self.nth += 1
            self.action = 0

            if self.nth == 21:
                obs_n = ['idle'] * self.n_players
            else:
                for i in range(self.n_players):
                    obs_n.append(self.players[i].prepare_obs())

        reward_n = [0] * self.n_players
        done_n = [0] * self.n_players
        info_n = None

        if nth==20 and naction==3:
            return self.end()
        return obs_n, reward_n, done_n, info_n, nth, naction

    def get_wonders(self):
        with open('v2_wonders.json', 'r') as f:
            wonders = json.load(f)
        self.wonders = [Wonder.from_dict(wonders.pop(random.randrange(len(wonders)))) \
                        for _ in range(self.n_players)]
        random.shuffle(self.wonders)

    def build_deck(self):
        with open('v2_cards.json', 'r') as f:
            cards = json.load(f)
        for i in range(1, 4):
            self.cards[i] = reduce(list.__add__, [[Card.from_dict(card)] \
                                                  * card['countPerNbPlayer'][str(self.n_players)] \
                                                  for card in cards[f'age{i}']['cards']]) + (i == 3) * \
                            random.sample([Card.from_dict(card) for card in cards['guildCards']], self.n_players + 2)
            random.shuffle(self.cards[i])

    def create_players(self):
        self.players = [DQNPlayer(env=self) for _ in range(self.n_players)]
        for i, player in enumerate(self.players):
            player.board = Board()
            player.board.wonder_to_build = self.wonders[i].stages
            player.board.wonder_name = self.wonders[i].name
            player.board.wonder_id = self.wonders[i].stages_id
        for i in range(self.n_players):
            self.players[i].left = self.players[i - 1].board
            self.players[i].right = self.players[i - (self.n_players - 1)].board
