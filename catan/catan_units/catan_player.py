from dataclasses import dataclass, field
from typing import List
from .catan_hex import CatanCorner, CatanEdge
from .game_parameters import *

@dataclass
class CatanPlayer:
    name: str
    color: str
    resource: dict = field(default_factory=lambda: {'resource': 0,
                                                    'ore': 0,
                                                    'brick': 0,
                                                    'lumber': 0,
                                                    'grain': 0,
                                                    'wool': 0})
    development: dict = field(default_factory=lambda:{'knight': 0,
                                                        'plenty': 0,
                                                        'road': 0,
                                                        'monopoly': 0,
                                                        'victory': 0,
                                                        'development': 0
                                                    })
    tokens: dict = field(default_factory=lambda:{'settlement': 5, 'city': 4, 'road': 15})
    trade_rates: dict = field(default_factory=lambda:{resource: 4 for resource in resource_types})

    settlements: List[CatanCorner] = field(default_factory=list)
    cities: List[CatanCorner] = field(default_factory=list)
    roads: List[CatanEdge] = field(default_factory=list)
    inactive_development: List[str] = field(default_factory=list)
    points: int = 0
    army: int = 0
    longest_road: bool = False
    largest_army: bool = False