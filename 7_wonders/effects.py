from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Effect:
    @staticmethod
    def translate_production(resources):
        resources = resources.replace('C', 'B')
        if resources.find('/') > -1:
            return {resources: 1}
        else:
            out = {}
            for r in resources:
                out[r] = out.get(r, 0) + 1
            return out

    @classmethod
    def from_dict(cls, e_dict):
        for key, value in e_dict.items():
            break
        return globals()[key[0].upper() + key[1:]](**value) if isinstance(value, dict) \
            else globals()[key[0].upper() + key[1:]](value)


@dataclass(init=False)
class Production(Effect):
    resources: dict = None
    isSellable: bool = True

    def __init__(self, resources, isSellable):
        self.resources = self.translate_production(resources)
        self.isSellable = isSellable

    def apply(self, board, *args):
        for r in self.resources.keys():
            board.production[r] += self.resources[r]
            if self.isSellable:
                board.sellable[r] += self.resources[r]


@dataclass
class Discount(Effect):
    resourceTypes: str = None
    providers: List[str] = field(default_factory=lambda: [])
    discountedPrice: int = 1

    def apply(self, board, *args):
        if self.resourceTypes == 'LGP':
            board.discount[2] = 1
        else:
            if self.providers == 'LEFT_PLAYER':
                board.discount[0] = 1
            else:
                board.discount[1] = 1


@dataclass
class Gold(Effect, int):
    x: int

    def apply(self, board, *args):
        board.coins += self


@dataclass
class Points(Effect, int):
    x: int

    def apply(self, board, *args):
        board.points += self


@dataclass
class Science(Effect, str):
    x: str

    def apply(self, board, *args):
        board.science[self.x.lower()] += 1


@dataclass
class Military(Effect, int):
    x: int

    def apply(self, board, *args):
        board.army += self


@dataclass
class PerBoardElement(Effect):
    boards: List[str] = field(default_factory=lambda: [])
    type: str = None
    gold: Optional[int] = 0
    points: Optional[int] = 0
    colors: Optional[List[str]] = field(default_factory=lambda: [])

    def apply(self, board, left, right, name):
        if self.gold > 0:
            for b in self.boards:
                if b == 'SELF':
                    b = 'board'
                if self.type == 'BUILT_WONDER_STAGES':
                    board.coins += locals()[b.lower()].wonders_built * self.gold
                elif self.type == 'COMPLETED_WONDER':
                    board.coins += (len(locals()[b.lower()].wonder_to_build) == 0) * self.gold
                elif self.type == 'CARD':
                    for c in self.colors:
                        board.coins += locals()[b.lower()].colors[c.lower()] * self.gold
        if self.points > 0:
            board.guilds[name.lower()] = True


@dataclass
class Action(Effect, str):
    x: str
