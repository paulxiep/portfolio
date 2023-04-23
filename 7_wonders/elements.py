import random
from dataclasses import dataclass, field
from functools import reduce
from itertools import product, combinations
from typing import Optional, List

import numpy as np
from gymnasium import Env

from effects import Effect
from src.constants import *


class BuildOption:
    '''
    to share requirement parsing method between cards and wonder stages
    '''

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


@dataclass
class Stage(BuildOption):
    requirements: dict
    effects: List[Effect]
    '''
    for wonder stage
    '''

    @classmethod
    def from_dict(cls, stage):
        return Stage(cls.translate_requirements(stage['requirements']),
                     [Effect.from_dict({k: v}) for k, v in stage['effects'].items()])


@dataclass
class Card(BuildOption):
    name: str
    color: str
    effect: Effect
    requirements: dict = field(default_factory=lambda: {})
    chain_parent: Optional[str] = None
    chain_children: list = field(default_factory=lambda: [])

    @classmethod
    def from_dict(cls, card):
        return Card(card['name'],
                    card['color'],
                    Effect.from_dict(card['effect']),
                    cls.translate_requirements(card.get('requirements', None)),
                    card.get('chainParent', None),
                    card.get('chainChildren', []))

    def apply(self, board, left, right):
        board.chains += self.chain_children
        self.effect.apply(board, left, right, self.name)


@dataclass
class Wonder:
    name: str
    side: str
    resource: str
    stages: list
    stages_id: list

    @classmethod
    def from_dict(cls, wonder_json, side=None):
        name = wonder_json['name']
        if side is None:
            choice = random.choice(['A', 'B'])
        else:
            choice = side
        side = wonder_json['sides'][choice]
        resource = side['initialResource']
        stages = side['stages']
        stages_id = list(map(lambda x: wonder_dict[(name, choice, x)], range(1, len(stages) + 1)))
        return Wonder(name, choice, resource, [Stage.from_dict(stage) for stage in stages], stages_id + [99])


@dataclass
class Board:
    # decided to make new resource type for each choice type
    production: dict = field(default_factory=lambda: {r: 0 for r in resources + choice_resources})
    sellable: dict = field(default_factory=lambda: {r: 0 for r in resources + choice_resources})
    chains: list = field(default_factory=lambda: [])
    coins: int = 3
    army: int = 0
    points: int = 0  # any flat points from Blue + Wonder
    guild_points: int = 0  # includes yellow guild-like points
    military_points: int = 0
    science_points: int = 0  # used to track science reward when training model, otherwise use calculate_science()
    discount: list = field(default_factory=lambda: [0, 0, 0])
    science: dict = field(default_factory=lambda: {'wheel': 0, 'compass': 0, 'tablet': 0, 'any': 0})
    guilds: dict = field(default_factory=lambda: {g: False for g in guilds})
    effects: List[Effect] = field(default_factory=lambda: [])
    wonder_effects: dict = field(default_factory=lambda: {a: False for a in action})
    colors: dict = field(default_factory=lambda: {'brown': 0, 'grey': 0, 'yellow': 0,
                                                  'blue': 0, 'green': 0, 'red': 0,
                                                  'purple': 0})
    wonder_to_build: List[Stage] = field(default_factory=lambda: [])
    wonder_built: int = 0
    wonder_name: str = None
    wonder_side: str = None
    wonder_id: List[int] = field(default_factory=lambda: [])
    built: dict = field(default_factory=lambda: {})

    def apply_card(self, card, left, right):
        self.built[card.name] = True
        self.colors[card.color.lower()] += 1
        card.apply(self, left, right)

    def build_wonder(self):
        self.wonder_built += 1
        self.wonder_id.pop(0)
        for effect in self.wonder_to_build.pop(0).effects:
            effect.apply(self)

    def to_obs(self):
        return np.concatenate((
            np.array(list(self.production.values())) / 10,
            np.array(list(self.sellable.values())) / 10,
            np.array([self.coins]) / 50,
            np.array([self.army]) / 20,
            np.array([self.points + self.military_points]) / 100,
            np.array([self.wonder_built]) / 4,
            np.array(list(self.colors.values())) / 10,
            np.array(list(self.science.values())) / 8,
            np.array([int(k in self.chains) for k in chain_dict.keys()]),
            np.array(self.discount),
            np.array(list(self.guilds.values())).astype(int),
            np.array(list(self.wonder_effects.values())).astype(int),
            np.array([wonder_name == self.wonder_name for wonder_name in wonder_names]),
            np.array(['B' == self.wonder_side])
            # below line kept in case old input is ever needed
            # np.array([int(v == self.wonder_id[0]) for v in wonder_dict.values()])
        ))

    def production_choices(self, neighbor=False):
        def add_resource(x, production):
            p = production.copy()
            p[x] += 1
            return p

        available = 'production' * (not neighbor) + 'sellable' * neighbor
        production_c = [{k: v for k, v in getattr(self, available).items() if k in resources}]
        choices = reduce(list.__add__, [[k] * v for k, v in getattr(self, available).items() if k in choice_resources])
        while len(choices) > 0:
            choice = choices.pop().split('/')
            production_c = reduce(list.__add__,
                                  map(lambda p: list(map(lambda x: add_resource(x, p), choice)), production_c))
        return production_c

    def calculate_guilds(self, left, right):
        for effect in self.effects:
            effect.apply_final(self, left, right)

    @staticmethod
    def calculate_science_set(science_set):
        return 7 * min([science_set[s] for s in sciences]) + reduce(int.__add__,
                                                                    [science_set[s] ** 2 for s in sciences])

    def calculate_science(self):
        def add_wildcard(tup):
            static = tup[0].copy()
            for s in tup[1]:
                static[s] += 1
            return static

        if self.science['any'] == 0:
            return self.calculate_science_set(self.science)
        else:
            static_set = [{k: v for k, v in self.science.items() if k != 'any'}]
            any_set = self.science['any'] * [sciences]
            if len(any_set) > 1:
                any_set = list(product(*any_set))
            else:
                any_set = [(s,) for s in sciences]
            return max(map(self.calculate_science_set, map(add_wildcard, product(static_set, any_set))))


@dataclass
class Player:
    board: Optional[Board] = None
    left: Optional[Board] = None
    right: Optional[Board] = None
    cards: Optional[List[Card]] = field(default_factory=lambda: [])
    hand: Optional[List[Card]] = field(default_factory=lambda: [])
    chosen: Optional[Card] = None
    action: Optional[str] = None

    def apply_card(self, card):
        self.board.apply_card(card, self.left, self.right)

    def build_wonder(self):
        self.board.build_wonder()

    def buildable(self, cost, name=None):
        '''
        determine if a card or wonder stage is buildable
        for card, pass a name argument to check for chains
        for wonder, pass None as name
        '''
        if name is not None:
            if name in self.board.chains:
                return True, (True, None)
        production_choices = self.board.production_choices()
        if 'C' in cost.keys():  # coin cost
            if self.board.coins >= cost['C']:
                return True, (False, [(0, 0, cost['C'])])
            else:
                return False, None
        for production_choice in production_choices:
            if all([production_choice[k] >= cost[k] for k in cost.keys()]):
                return True, (True, None)
        buy_choices_left = self.left.production_choices(neighbor=True)
        buy_choices_right = self.right.production_choices(neighbor=True)
        pay_choices = set()
        for production_choice in production_choices:
            pay_choice = []
            for k in cost.keys():
                p = production_choice[k]
                c = cost[k]
                if k in ['W', 'S', 'O', 'B']:
                    lc = 2 - self.board.discount[0]
                    rc = 2 - self.board.discount[1]
                else:
                    lc = 2 - self.board.discount[2]
                    rc = 2 - self.board.discount[2]
                if p < c:
                    deficit = c - p
                    bls = [bc[k] for bc in buy_choices_left]
                    brs = [bc[k] for bc in buy_choices_right]
                    partial_choice = set()
                    for bl, br in product(bls, brs):
                        if bl + br > deficit:
                            pool = ['l'] * bl + ['r'] * br
                            comb = combinations(pool, deficit)
                            partial_choice = partial_choice.union(
                                set(map(lambda x: (x.count('l') * lc, x.count('r') * rc), comb)))
                        elif bl + br == deficit:
                            partial_choice = partial_choice.union({(bl * lc, br * rc)})
                        else:
                            partial_choice = partial_choice.union({(999, 999)})
                    pay_choice.append(partial_choice)
            pay_set = set()
            for item in product(*pay_choice):
                pay_set = pay_set.union({reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]), item)})
            pay_choices = pay_choices.union(set(filter(lambda x: x[0] + x[1] <= self.board.coins, pay_set)))
        if len(pay_choices) > 0:
            return True, (False, list(pay_choices))
        else:
            return False, None

    def calculate_guilds(self):
        '''
        call once before calculating end game points
        will add guild points (including yellow guild-like points) to self.guild_points
        '''
        self.board.calculate_guilds(self.left, self.right)

    def calculate_science(self):
        return self.board.calculate_science()


@dataclass(init=False)
class AIPlayer(Player):
    env: Env = field(compare=False, default=None)
    model = None
    explore = None

    def __init__(
            self,
            *args,
            env=None,
            model=None,
            explore=None
    ):
        super().__init__(*args)
        self.env = env
        self.model = model
        if explore is None:
            self.explore = random.choice([True, True, True, False])
        else:
            self.explore = explore

    @staticmethod
    def match_payment(cost, out):
        '''
        legacy + possible future method
        will be used in case of regression model in payment phase,
        to match model output to actual valid payment options
        '''
        def match_ratio(x, out):
            out = out[0] + 1, out[1] + 1
            if out[1] == 0:
                out = out[0], out[1] + 0.01
            if out[0] == 0:
                out = out[0] + 0.01, out[1]
            x = x[0] + 1, x[1] + 1
            return abs(x[0] / x[1] - out[0] / out[1]) + abs(x[1] / x[0] - out[1] / out[0])

        return min(cost, key=lambda x: match_ratio(x, out))

    def prepare_obs(self):
        '''
        generate observation
        '''
        if self.env.action in [0, 3]:
            cards = np.zeros(80)
            np.add.at(cards, [card_dict[card.name] for card in self.cards], 1)
        elif self.env.action in [1, 2]:
            cards = np.zeros(80)
            np.add.at(cards, [card_dict[self.chosen.name]], 1)
        else:
            cards = np.zeros(80)
        if self.env.action == 3:
            if self.board.wonder_effects['PLAY_DISCARDED'] and len(self.env.discarded) > 0:
                cur_action = 0
            else:
                cur_action = 3
        else:
            cur_action = self.env.action
        if self.env.nth in [6, 13, 20] and not self.board.wonder_effects['PLAY_LAST_CARD'] and self.env.action != 3:
            cur_action = 3
        if cur_action == 3:
            return 'idle'
        else:
            return np.concatenate((
                cards,
                self.board.to_obs(),
                self.left.to_obs(),
                self.right.to_obs(),
                np.array([len(self.env.discarded)]) / 20,
                np.array([self.env.nth / 21]),
                np.array([int(self.env.nth in range(7, 14))])
            )).astype(float)

    def generate_mask(self, card_array):
        '''
        for ineligible move masking from model output
        '''
        reverse_card_dict = {v: k for k, v in card_dict.items()}

        def buildable(card_id_plus, free_color=False):
            card_id_plus = int(card_id_plus)
            if card_id_plus == 0:
                return 0
            card_name = reverse_card_dict[card_id_plus - 1]
            for card in self.cards:
                if card.name == card_name:
                    if (not free_color) or self.board.colors[card.color.lower()] > 0:
                        return int(
                            not self.board.built.get(card.name, False) and \
                                (self.board.wonder_effects['PLAY_DISCARDED'] or \
                                     self.buildable(card.requirements, card.name)[
                                0]))
                    else:
                        return 2

        if self.board.wonder_id[0] == 99 or not \
                self.buildable(self.board.wonder_to_build[0].requirements, None)[0]:
            wonder_buildable = 0
        else:
            wonder_buildable = 1
        discard_chance = 1
        if (self.env.nth in [0, 7, 14] and self.board.wonder_effects['FIRST_FREE_PER_AGE']) or \
                (self.env.nth in [5, 12, 19] and self.board.wonder_effects['LAST_FREE_PER_AGE']):
            wonder_buildable = 0
            discard_chance = 0
            card_mask = card_array
        elif self.board.wonder_effects['FIRST_FREE_PER_COLOR']:
            card_mask = np.vectorize(lambda x: buildable(x, True))(np.arange(80)
                                                                   * card_array.astype(bool)
                                                                   + card_array.astype(bool))
        else:
            if self.board.wonder_effects['PLAY_DISCARDED']:
                discard_chance = 0
                wonder_buildable = 0
            card_mask = np.vectorize(lambda x: buildable(x, False))(
                np.arange(80) * card_array.astype(bool) + card_array.astype(bool))
        wonder_mask = wonder_buildable * card_array
        if not self.board.wonder_effects['FIRST_FREE_PER_COLOR']:
            return np.concatenate((card_mask, wonder_mask, card_array * discard_chance))
        else:
            return np.concatenate(
                (card_mask.astype(bool).astype(int), wonder_mask, card_array - (card_mask == 2).astype(int)))


@dataclass(init=False)
class DQPlayer(AIPlayer):
    play = None

    def __init__(
            self,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.play = None

    def select_action(self, obs, explore=False):
        def random_nonzero(value):
            if value == 0:
                return value
            else:
                return random.random()

        def remove_zero(value):
            if value == 0:
                return -999999
            else:
                return value

        if self.env.nth not in [6, 13, 20] or self.board.wonder_effects['PLAY_LAST_CARD'] or self.board.wonder_effects['PLAY_DISCARDED']:
            if (self.env.action == 0 and (self.env.nth not in [6, 13, 20] or self.board.wonder_effects['PLAY_LAST_CARD'])) \
                    or (self.env.action == 3 and self.env.nth not in [5, 12, 19] \
                                        and self.board.wonder_effects['PLAY_DISCARDED'] and len(
                    self.env.discarded) > 0):
                mask = self.generate_mask(obs[:80])
                if self.model is not None and \
                        (not explore or (not self.explore and random.random() < 0.9) or random.random() < 0.5):
                    out = mask * self.model(np.expand_dims(obs, axis=0)).numpy()[0]
                    iarg = np.argmax(np.vectorize(remove_zero)(out))
                    icard = iarg % 80
                    self.play = iarg // 80
                    return icard, None, mask
                else:
                    if not explore:
                        if mask[:160].sum() > 0:
                            iarg = np.argmax(np.vectorize(random_nonzero)(mask[:160]))
                            self.play = iarg // 80
                            icard = iarg % 80
                        else:
                            icard = np.argmax(np.vectorize(random_nonzero)(obs[:80]))
                            self.play = 2
                    else:
                        icard = np.argmax(np.vectorize(random_nonzero)(obs[:80]))
                        self.play = None
                return icard, None, mask

            elif self.env.action == 1 and (self.env.nth not in [6, 13, 20] or self.board.wonder_effects['PLAY_LAST_CARD']):
                if not (self.env.nth in [0, 7, 14] and self.board.wonder_effects['FIRST_FREE_PER_AGE']) and \
                        not (self.env.nth in [5, 12, 19] and self.board.wonder_effects['LAST_FREE_PER_AGE']) and \
                        not (self.board.wonder_effects['FIRST_FREE_PER_COLOR'] and self.board.colors[
                            self.chosen.color.lower()] == 0):

                    if self.play is None:
                        wonder_possible = False
                        card_possible = False
                        if not self.board.wonder_id[0] == 99 and \
                                self.buildable(self.board.wonder_to_build[0].requirements, None)[0]:
                            wonder_possible = True
                        elif not self.board.built.get(self.chosen.name, False) and \
                                self.buildable(self.chosen.requirements, self.chosen.name)[0]:
                            card_possible = True
                        else:
                            return 2, None, None
                        if wonder_possible and card_possible:
                            return random.choice[0, 1, 1], None, None
                        elif card_possible:
                            return 0, None, None
                        else:
                            return 1, None, None
                    else:
                        return self.play, None, None
                else:
                    if self.play is None:
                        if not self.board.built.get(self.chosen.name, False):
                            return 0, None, None
                        else:
                            return 2, None, None
                    else:
                        return self.play, None, None

            elif self.env.action == 2 and (self.env.nth not in [6, 13, 20] or self.board.wonder_effects['PLAY_LAST_CARD']):
                if self.action == 0:  # card
                    if not (self.env.nth in [0, 7, 14] and self.board.wonder_effects['FIRST_FREE_PER_AGE']) and \
                            not (self.env.nth in [5, 12, 19] and self.board.wonder_effects['LAST_FREE_PER_AGE']) and \
                            not (self.board.wonder_effects['FIRST_FREE_PER_COLOR'] and self.board.colors[
                                self.chosen.color.lower()] == 0):
                        cost = self.buildable(self.chosen.requirements, self.chosen.name)[1]
                    else:
                        cost = True, None
                elif self.action == 1:  # wonder
                    cost = self.buildable(self.board.wonder_to_build[0].requirements, None)[1]
                else:  # discard for 3 coins
                    return 'idle', None, None
                if cost[0]:
                    return [0, 0, 0], None, None
                else:
                    cost = cost[1]
                    if cost[0][0] == 0 and cost[0][1] == 0:
                        return [0, 0, cost[0][2] / 20], None, None
                    return (np.array(
                        list(min(cost, key=lambda x: x[0] + x[1] + random.random())) + [0]) / 20).tolist(), None, None

            else:
                return 'idle', None, None
        else:
            return 'idle', None, None
