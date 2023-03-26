import json
import random
from functools import reduce
from card import Card

class Game:
    def __init__(self, players):
        self.players = players
        self.cards = {1: [], 2: [], 3: []}

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

    def build_deck(self):
        with open('v2_cards.json', 'r') as f:
            cards = json.load(f)
        for i in range(1, 4):
            self.cards[i] = reduce(list.__add__, [[Card(card['name'], card['color'], card['effect'],
                                  self.translate_requirements(card.get('requirements', None)),
                                  card.get('chainParent', None),
                                  card.get('chainChildren', []))] \
                                 * card['countPerNbPlayer'][str(self.players)] \
                                 for card in cards[f'age{i}']['cards']]) + (i==3) * \
                                    random.sample([Card(card['name'], card['color'], card['effect'],
                                  self.translate_requirements(card.get('requirements', None)),
                                  card.get('chainParent', None),
                                  card.get('chainChildren', [])) for card in cards['guildCards']], self.players+2)