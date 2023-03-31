import random
from dataclasses import dataclass, field
from typing import Optional, List

from effects import Effect


class BuildOption:
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


@dataclass
class Wonder:
    name: str
    resource: str
    stages: list

    @classmethod
    def from_dict(cls, wonder_json):
        name = wonder_json['name']
        side = wonder_json['sides'][random.choice(['A', 'B'])]
        resource = side['initialResource']
        stages = side['stages']
        return Wonder(name, resource, [Stage.from_dict(stage) for stage in stages])


@dataclass
class Board:
    # decided to make new resource type for each choice type
    production: dict = field(default_factory=lambda: {})
    sellable: dict = field(default_factory=lambda: {})
    chains: list = field(default_factory=lambda: [])
    coins: int = 3
    army: int = 0
    points: int = 0
    discount: list = field(default_factory=lambda: [])
    science: dict = field(default_factory=lambda: {'wheel': 0, 'compass': 0, 'tablet': 0, 'any': 0})
    guilds: list = field(default_factory=lambda: [])
    colors: list = field(default_factory=lambda: {'brown': 0, 'grey': 0, 'yellow': 0,
                                                  'blue': 0, 'green': 0, 'red': 0,
                                                  'purple': 0})
    wonder_to_build: list = field(default_factory=lambda: [])
    wonder_built: int = 0

    def place_card(self, card):
        pass

    def build_wonder(self):
        pass


@dataclass
class Player:
    board: Optional[Board] = None
    left: Optional[Board] = None
    right: Optional[Board] = None
    cards: Optional[List[Card]] = None
    chosen: Optional[Card] = None
    clockwise: bool = True

    def make_obs(self, action=None):
        pass
