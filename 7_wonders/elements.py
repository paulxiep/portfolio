from dataclasses import dataclass, field
from gymnasium.spaces import Sequence, Dict
from typing import Optional, List
from effects import *
import random

@dataclass
class Stage:
    requirements: dict
    effect: Effect

    @classmethod
    def from_dict(cls, stage):
        return Stage(stage['requirements'], Effect.from_dict(stage['effects']))

@dataclass
class Card:
    name: str
    color: str
    effect: Effect
    requirements: dict = field(default_factory=lambda: {})
    chain_parent: Optional[str] = None
    chain_children: list = field(default_factory=lambda: [])

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

