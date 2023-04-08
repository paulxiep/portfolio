import random
from dataclasses import dataclass, field
from functools import reduce
from itertools import product, combinations
from typing import Optional, List

import numpy as np
from gymnasium import Env

from effects import Effect
from src.constants import *
from src.model import DQNetwork


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
    resource: str
    stages: list
    stages_id: list

    @classmethod
    def from_dict(cls, wonder_json):
        name = wonder_json['name']
        choice = random.choice(['A', 'B'])
        side = wonder_json['sides'][choice]
        resource = side['initialResource']
        stages = side['stages']
        stages_id = list(map(lambda x: wonder_dict[(name, choice, x)], range(1, len(stages) + 1)))
        return Wonder(name, resource, [Stage.from_dict(stage) for stage in stages], stages_id + [99])


@dataclass
class Board:
    # decided to make new resource type for each choice type
    production: dict = field(default_factory=lambda: {r: 0 for r in resources + choice_resources})
    sellable: dict = field(default_factory=lambda: {r: 0 for r in resources + choice_resources})
    chains: list = field(default_factory=lambda: [])
    coins: int = 3
    army: int = 0
    points: int = 0
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
    wonder_id: List[int] = field(default_factory=lambda: [])
    built: list = field(default_factory=lambda: [])

    def apply_card(self, card, left, right):
        self.built.append(card.name)
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
            np.array([self.points]) / 100,
            np.array([self.wonder_built]) / 4,
            np.array(list(self.colors.values())) / 10,
            np.array(list(self.science.values())) / 8,
            np.array([int(k in self.chains) for k in chain_dict.keys()]),
            np.array(self.discount),
            np.array(list(self.guilds.values())).astype(int),
            np.array(list(self.wonder_effects.values())).astype(int),
            np.array([int(v == self.wonder_id[0]) for v in wonder_dict.values()])
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
            if isinstance(tup[1], str):
                static[tup[1]] += 1
            else:
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

    def buildable(self, cost, name):
        if name is not None:
            if name in self.board.chains:
                return True, (True, None)

        production_choices = self.board.production_choices()
        if 'C' in cost.keys():
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
        self.board.calculate_guilds(self.left, self.right)

    def calculate_science(self):
        return self.board.calculate_science()


@dataclass(init=False)
class DQPlayer(Player):
    env: Env = field(compare=False, default=None)
    model = None

    def __init__(
            self,
            *args,
            env=None,
            model=None
    ):
        super().__init__(*args)
        self.env = env
        if model is None:
            self.model = DQNetwork({'choose': 80, 'play': 3})
        else:
            self.model = model

    @staticmethod
    def match_payment(cost, out):
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
        if self.env.nth in [6, 13, 20] and not self.board.wonder_effects['PLAY_LAST_CARD']:
            cur_action = 3
        if cur_action == 3:
            return 'idle'
        else:
            out_action = np.zeros(3)
            np.add.at(out_action, [cur_action], 1)
            return np.concatenate((
                cards,
                self.board.to_obs(),
                self.left.to_obs(),
                self.right.to_obs(),
                np.array([self.env.nth / 21]),
                out_action,
                np.array([int(self.env.nth in range(7, 14))])
            )).astype(float)

    def select_action(self, obs, explore=False):
        def random_nonzero(value):
            if value == 0:
                return value
            else:
                return random.random()

        def remove_zero(value):
            if value == 0:
                return -999
            else:
                return value

        if self.env.nth not in [6, 13, 20] or self.board.wonder_effects['PLAY_LAST_CARD']:
            if self.env.action == 0 or (self.env.action == 3 and self.board.wonder_effects['PLAY_DISCARDED'] and len(
                    self.env.discarded) > 0):
                out = self.model(obs, 'choose', mask=obs[:80]).numpy()
                if not explore or random.random() < 0.1:
                    iarg = np.argmax(np.vectorize(remove_zero)(out[0]))
                else:
                    iarg = np.argmax(np.vectorize(random_nonzero)(out[0]))
                return iarg, out[0][iarg], obs[:80]
                # return np.random.choice(range(80), p=out[0])
            elif self.env.action == 1 and \
                    not (self.env.nth in [0, 7, 14] and self.board.wonder_effects['FIRST_FREE_PER_AGE']) and \
                    not (self.env.nth in [5, 12, 19] and self.board.wonder_effects['LAST_FREE_PER_AGE']) and \
                    not (self.board.wonder_effects['FIRST_FREE_PER_COLOR'] and self.board.colors[
                        self.chosen.color.lower()] == 0):
                mask = np.ones(3)
                if not self.buildable(self.chosen.requirements, self.chosen.name)[0]:
                    mask[0] = 0
                if self.board.wonder_id[0] == 99 or not \
                        self.buildable(self.board.wonder_to_build[0].requirements, None)[0]:
                    mask[1] = 0
                # if mask[0] == 1 or mask[1] == 1:
                #     mask[2] = 0
                if mask[0] == 1 and mask[1] == 1:
                    if random.random() < 0.8:
                        return 0, None, None
                    else:
                        return 1, None, None
                if mask[0]:
                    return 0, None, None
                if mask[1]:
                    return 1, None, None
                return 2, None, None
                # out = self.model(obs, 'play', mask=mask).numpy()
                # if not explore or random.random() < 0.1:
                #     iarg = np.argmax(np.vectorize(remove_zero)(out[0]))
                # else:
                #     iarg = np.argmax(np.vectorize(random_nonzero)(out[0]))
                # return iarg, out[0][iarg], mask
                # return np.random.choice(range(3), p=out[0])
            elif self.env.action == 2 and \
                    not (self.env.nth in [0, 7, 14] and self.board.wonder_effects['FIRST_FREE_PER_AGE']) and \
                    not (self.env.nth in [5, 12, 19] and self.board.wonder_effects['LAST_FREE_PER_AGE']) and \
                    not (self.board.wonder_effects['FIRST_FREE_PER_COLOR'] and self.board.colors[
                        self.chosen.color.lower()] == 0):

                if self.action == 0:  # card
                    cost = self.buildable(self.chosen.requirements, self.chosen.name)[1]
                elif self.action == 1:  # wonder
                    cost = self.buildable(self.board.wonder_to_build[0].requirements, None)[1]
                else:  # discard for 3 coins
                    return 'idle', None, None
                if cost[0]:
                    return [0, 0, 0], None, None
                else:
                    cost = cost[1]
                    if cost[0] == 0 and cost[1] == 0:
                        # cost is coin
                        return [0, 0, cost[2] / 20], None, None
                    return (np.array(list(random.choice(cost)) + [0]) / 20).tolist(), None, None
                    # out = self.model(obs, 'pay')[0].numpy()
                    # return np.array(list(self.match_payment(cost, out)) + [0])/20
            else:
                return 'idle', None, None
        else:
            return 'idle', None, None
