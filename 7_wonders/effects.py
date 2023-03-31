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
    resources: str = None
    isSellable: bool = True

    def __init__(self, resources, isSellable):
        self.resources = self.translate_production(resources)
        self.isSellable = isSellable


@dataclass
class Discount(Effect):
    resourceTypes: str = None
    providers: List[str] = field(default_factory=lambda: [])
    discountedPrice: int = 1


@dataclass
class Gold(Effect, int):
    x: int


@dataclass
class Points(Effect, int):
    x: int


@dataclass
class Science(Effect, str):
    x: str


@dataclass
class Military(Effect, int):
    x: int


@dataclass
class PerBoardElement(Effect):
    boards: List[str] = field(default_factory=lambda: [])
    type: str = None
    gold: Optional[int] = 0
    points: Optional[int] = 0
    colors: Optional[List[str]] = field(default_factory=lambda: [])


@dataclass
class Action(Effect, str):
    x: str
