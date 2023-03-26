from dataclasses import dataclass, field

@dataclass
class Card:
    name: str
    color: str
    effect: dict
    requirements: dict = field(default_factory=lambda: {})
    chain_parent: str = None
    chain_children: list = field(default_factory=lambda: [])