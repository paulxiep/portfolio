import json
import random
from functools import reduce

import numpy as np
import pandas as pd
from gymnasium import Env
from gymnasium.spaces import MultiBinary, Box, Tuple, Discrete
from gymnasium.spaces.utils import flatten_space

from elements import Card, Wonder, Board
from src.constants import *
from src.utils import *


class Game(Env):
    def __init__(self, players, id=0):
        super().__init__()
        '''
        actual 7 wonders supports up to 7 players, but for the end of phase 1, 
        this class will support up to 6 players for now, with Helicarnassus removed from play,
        as potentially the current implementation of Helicarnassus' special effect is wrong
        '''
        self.id = id
        self.n_players = len(players)
        self.players = players
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

        # note that observation space defined here still has 3 units more length
        # than outputted from Player class. I still fail to find the source of the discrepancy.
        self.observation_space = flatten_space(Tuple((MultiBinary(80), board_space, board_space, board_space,
                                                      Box(np.zeros(1), np.ones(1)),
                                                      # game progress; order of card in play
                                                      Discrete(3),  # choose, play for price, pay, play_discarded/idle
                                                      Discrete(1)  # reverse passing direction
                                                      )))

    def setup(self):
        self.build_deck()
        self.get_wonders()
        self.assign_players()
        self.nth = 0
        self.action = 0
        self.memory = []
        self.discarded = []
        self.era = 1
        for i in range(self.n_players):
            self.players[i].hand = self.cards[self.era][(i * 7): ((i + 1) * 7)].copy()
            self.players[i].cards = self.cards[self.era][(i * 7): ((i + 1) * 7)].copy()

    def run_and_save(self):
        '''
        not meant to be used, for testing memory data and new weight performance
        '''
        self.collect(False)
        chooses = []
        # plays = []
        for i in range(self.n_players):
            choose = squash_idle(list(pd.DataFrame(self.memory)[i]))
            chooses.append(choose)
        points = []
        for i, choose in enumerate(chooses):
            points.append(choose[-1][5])
        return np.array(points)

    def run_and_return(self):
        '''
        for use by the dq_trainer to generate game data
        '''
        self.collect(True)
        chooses = []
        for i in range(self.n_players):
            choose = squash_idle(list(pd.DataFrame(self.memory)[i]))
            chooses.append(choose)
        return chooses

    def end(self, reward_n):
        '''
        calculate reward (points) at game end
        '''
        obs_n = ['idle'] * self.n_players
        done_n = [1] * self.n_players
        info_n = None
        reward_n_out = reward_n.copy()
        for i in range(self.n_players):
            self.players[i].calculate_guilds()
            own_points = (
                self.players[i].board.coins // 3 +
                self.players[i].board.points +
                self.players[i].board.guild_points +
                self.players[i].board.military_points +
                self.players[i].calculate_science()
            )
            reward_n_out[i] += own_points
        return obs_n, reward_n_out, done_n, info_n, 21, 0

    def collect(self, training):
        '''
        for simulating whole game
        '''
        done = False
        obs_n = list(map(lambda i: self.players[i].prepare_obs(), range(self.n_players)))
        while not done:
            action_n, raw_n, mask_n = tuple(
                zip(*map(lambda i: self.players[i].select_action(obs_n[i], training), range(self.n_players))))
            next_obs_n, reward_n, done_n, info_n, nth, action = self.step(action_n)
            self.memory.append(
                [(obs_n[i], action_n[i], next_obs_n[i], reward_n[i], done_n[i],
                  self.players[i].board.coins // 3 +
                  self.players[i].board.points +
                  self.players[i].board.guild_points +
                  self.players[i].board.military_points +
                  self.players[i].calculate_science(),
                  self.players[i].calculate_science(),
                  self.players[i].board.points,
                  mask_n[i], nth, action) for i
                 in range(self.n_players)])
            obs_n = next_obs_n
            done = bool(done_n[0])

    def turn(self, training):
        '''
        not used yet, for processing single turn
        '''
        obs_n = list(map(lambda i: self.players[i].prepare_obs(), range(self.n_players)))
        action_n, raw_n, mask_n = tuple(
            zip(*map(lambda i: self.players[i].select_action(obs_n[i], training), range(self.n_players))))
        next_obs_n, reward_n, done_n, info_n, nth, action = self.step(action_n)
        self.memory.append(
            [(obs_n[i], action_n[i], next_obs_n[i], reward_n[i], done_n[i], raw_n[i], mask_n[i], nth, action) for i in
             range(self.n_players)])

    def step(self, action_n):
        nth, naction = self.nth, self.action
        reward_n = [0] * self.n_players
        done_n = [0] * self.n_players
        info_n = None
        if False:  # self.nth == 20 and self.action == 3:
            pass  # return self.end()
        elif self.action == 0:
            self.action = 1
            obs_n = []
            for i in range(self.n_players):
                if self.nth not in [6, 13, 20] or self.players[i].board.wonder_effects['PLAY_LAST_CARD']:
                    chosen = {v: k for k, v in card_dict.items()}[action_n[i]]
                    # print(chosen, action_n[i], [card.name for card in self.players[i].hand])
                    for cid in range(len(self.players[i].hand)):
                        if self.players[i].hand[cid].name == chosen:
                            self.players[i].chosen = self.players[i].hand.pop(cid)
                            break
                    # print(self.nth, i, [c.name for c in self.players[i].hand])
            for i in range(self.n_players):
                if self.nth not in [6, 13, 20] or self.players[i].board.wonder_effects['PLAY_LAST_CARD']:
                    # self.players[i].cards = self.players[i].hand.copy()
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
                            (self.players[i].board.wonder_effects['FIRST_FREE_PER_COLOR'] and
                             self.players[i].board.colors[self.players[i].chosen.color.lower()] == 0):
                        # obs_n.append('idle')
                        self.players[i].action = action_n[i]
                        # pass
                    else:
                        self.players[i].action = action_n[i]
                        if action_n[i] == 2:
                            self.players[i].board.coins += 3
                            self.discarded.append(self.players[i].chosen)
            for i in range(self.n_players):
                if self.nth not in [6, 13, 20] or self.players[i].board.wonder_effects['PLAY_LAST_CARD']:
                    if (self.nth in [0, 7, 14] and self.players[i].board.wonder_effects['FIRST_FREE_PER_AGE']) or \
                            (self.nth in [5, 12, 19] and self.players[i].board.wonder_effects['LAST_FREE_PER_AGE']): #or \
                            # (self.players[i].board.wonder_effects['FIRST_FREE_PER_COLOR'] and
                            #  self.players[i].board.colors[self.players[i].chosen.color.lower()] == 0):
                        obs_n.append('idle')
                    else:
                        if action_n[i] == 2 or (action_n[i] == 0 and
                                                (self.players[i].board.wonder_effects['FIRST_FREE_PER_COLOR'] and
                                                self.players[i].board.colors[self.players[i].chosen.color.lower()] == 0)):
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
                    if (self.nth in [0, 7, 14] and self.players[i].board.wonder_effects['FIRST_FREE_PER_AGE']) or \
                            (self.nth in [5, 12, 19] and self.players[i].board.wonder_effects['LAST_FREE_PER_AGE']): #or \
                            # (self.players[i].board.wonder_effects['FIRST_FREE_PER_COLOR'] and
                            #  self.players[i].board.colors[self.players[i].chosen.color.lower()] == 0)):
                        # obs_n.append('idle')
                        pass
                    else:
                        if self.players[i].action != 2:
                            if action_n[i][0] == 0 and action_n[i][1] == 0:
                                self.players[i].board.coins -= int(round(action_n[i][2] * 20))
                            else:
                                self.players[i].board.coins -= int(round((action_n[i][0]) * 20) + round((action_n[i][1]) * 20))
                                self.players[i].left.coins += int(round(action_n[i][0] * 20))
                                self.players[i].right.coins += int(round(action_n[i][1] * 20))
                            if self.players[i].action == 0:
                                self.players[i].apply_card(self.players[i].chosen)
                            elif self.players[i].action == 1:
                                self.players[i].build_wonder()
                        if not self.players[i].board.wonder_effects['PLAY_DISCARDED']:
                            # obs_n.append('idle')
                            pass
                        else:
                            if len(self.discarded) > 0:
                                self.players[i].cards = self.discarded

                if (self.nth in [0, 7, 14] and self.players[i].board.wonder_effects['FIRST_FREE_PER_AGE']) or \
                        (self.nth in [5, 12, 19] and self.players[i].board.wonder_effects['LAST_FREE_PER_AGE']): #or \
                        # (self.players[i].board.wonder_effects['FIRST_FREE_PER_COLOR'] and self.players[i].board.colors[
                        #     self.players[i].chosen.color.lower()] == 0 and self.players[i].action == 0):
                    self.players[i].apply_card(self.players[i].chosen)

            if self.nth in [6, 13, 20]:  # discard at end of era
                for i in range(self.n_players):
                    # print(len(self.players[i].hand))
                    for card in self.players[i].hand:
                        self.discarded.append(card)

            for i in range(self.n_players):
                if self.nth not in [6, 13, 20] or self.players[i].board.wonder_effects['PLAY_LAST_CARD']:
                    if ((self.nth in [0, 7, 14] and self.players[i].board.wonder_effects['FIRST_FREE_PER_AGE']) or \
                            (self.nth in [5, 12, 19] and self.players[i].board.wonder_effects['LAST_FREE_PER_AGE']) or \
                            (self.players[i].board.wonder_effects['FIRST_FREE_PER_COLOR'] and
                             self.players[i].board.colors[self.players[i].chosen.color.lower()] == 0)):
                        obs_n.append('idle')
                    else:
                        if not self.players[i].board.wonder_effects['PLAY_DISCARDED']:
                            obs_n.append('idle')
                        else:
                            if len(self.discarded) > 0:
                                obs_n.append(self.players[i].prepare_obs())
                            else:
                                obs_n.append('idle')
                else:
                    obs_n.append('idle')

        elif self.action == 3:
            obs_n = []
            for i in range(self.n_players):
                if self.nth not in [6, 13, 20] or self.players[i].board.wonder_effects['PLAY_LAST_CARD']:
                    if self.players[i].board.wonder_effects['PLAY_DISCARDED']:
                        self.players[i].board.wonder_effects['PLAY_DISCARDED'] = False
                        if len(self.discarded) > 0:
                            chosen = {v: k for k, v in card_dict.items()}[action_n[i]]
                            for cid in range(len(self.players[i].cards)):
                                if self.players[i].cards[cid].name == chosen:
                                    self.players[i].chosen = self.players[i].cards.pop(cid)
                                    break
                            self.players[i].apply_card(self.players[i].chosen)
            if self.nth not in [5, 6, 12, 13, 19, 20]:  # not end of era
                if self.nth in range(7, 14):  # pass counter clockwise
                    buffer_from = self.players[-1].hand.copy()
                    for i in range(self.n_players):
                        buffer_to = self.players[i].hand.copy()
                        self.players[i].hand = buffer_from.copy()
                        self.players[i].cards = self.players[i].hand.copy()
                        buffer_from = buffer_to
                else:  # pass clockwise
                    buffer = self.players[0].hand.copy()
                    for i in range(self.n_players):
                        if i < self.n_players - 1:
                            self.players[i].hand = self.players[i + 1].hand.copy()
                        else:
                            self.players[i].hand = buffer.copy()
                        self.players[i].cards = self.players[i].hand.copy()
            elif self.nth in [6, 13, 20]:  # end of era
                for i in range(self.n_players):
                    for opponent in [self.players[i - 1], self.players[i - (self.n_players - 1)]]:
                        if self.players[i].board.army < opponent.board.army:
                            self.players[i].board.military_points -= 1
                        elif self.players[i].board.army > opponent.board.army:
                            self.players[i].board.military_points += 1 + (self.era - 1) * 2
                if self.nth == 20:  # end of game
                    pass
                else:  # start next era
                    if self.nth == 13:  # start 3rd era
                        self.era = 3
                    else:
                        self.era = 2
                    for i in range(self.n_players):
                        self.players[i].hand = self.cards[self.era][(i * 7): ((i + 1) * 7)].copy()
                        self.players[i].cards = self.cards[self.era][(i * 7): ((i + 1) * 7)].copy()
            else: # if play last card
                for i in range(self.n_players):
                    self.players[i].cards = self.players[i].hand.copy()


            self.nth += 1
            self.action = 0

            if self.nth == 21:
                obs_n = ['idle'] * self.n_players
            else:
                for i in range(self.n_players):
                    obs_n.append(self.players[i].prepare_obs())
            reward_n = []
            for i in range(self.n_players):
                new_science = self.players[i].board.calculate_science()-self.players[i].board.science_points
                self.players[i].board.science_points += new_science
                reward_n.append(new_science)

        if nth == 20 and naction == 3:
            return self.end(reward_n)
        return obs_n, reward_n, done_n, info_n, nth, naction

    def get_wonders(self):
        with open('v2_wonders.json', 'r') as f:
            wonders = json.load(f)
        wonders = wonders[:4] + wonders[5:7]
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

    def assign_players(self):
        for player in self.players:
            player.env = self
        for i, player in enumerate(self.players):
            player.board = Board()
            player.board.wonder_to_build = self.wonders[i].stages
            player.board.wonder_name = self.wonders[i].name
            player.board.wonder_side = self.wonders[i].side
            player.board.wonder_id = self.wonders[i].stages_id
        for i in range(self.n_players):
            self.players[i].right = self.players[i - 1].board
            self.players[i].left = self.players[i - (self.n_players - 1)].board
