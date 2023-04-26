from src.constants import card_dict
from elements import AIPlayer, Board, Wonder, Card
from effects import *
from typing import List
import json
import numpy as np

with open('7_wonders/v2_cards.json', 'r') as f:
    card_json = json.load(f)

with open('7_wonders/v2_wonders.json', 'r') as f:
    wonder_json = json.load(f)


@dataclass
class MockEnv:
    nth: int


@dataclass
class BoardData:
    wonder_name: str
    wonder_side: str
    wonder_stages_built: int
    built_structures: List[str]
    coins: int
    play_discarded: bool

    def init_wonder(self):
        for wonder in wonder_json:
            if wonder['name']==self.wonder_name:
                self.wonder = Wonder.from_dict(wonder, side=self.wonder_side)
                break

    def prepare_board(self):
        self.init_wonder()
        out = Board()
        out.coins = self.coins
        out.wonder_built = self.wonder_stages_built
        out.wonder_to_build = self.wonder.stages[self.wonder_stages_built:]
        out.wonder_name = self.wonder_name
        out.wonder_side = self.wonder_side
        out.wonder_id = self.wonder.stages_id[self.wonder_stages_built:]
        out.production[self.wonder.resource] += 1
        out.sellable[self.wonder.resource] += 1
        out.wonder_effects['PLAY_DISCARDED'] = self.play_discarded
        for i in range(self.wonder_stages_built):
            for effect in self.wonder.stages[i].effects:
                if not isinstance(effect, Gold):
                    effect.apply(out)
        for structure in self.built_structures:
            break_outside = False
            for age in [1, 2, 3]:
                for card_data in card_json[f'age{age}']['cards']:
                    if card_data['name'] == structure:
                        card = Card.from_dict(card_data)
                        if not isinstance(card.effect, PerBoardElement) and not isinstance(card.effect, Gold):
                            out.apply_card(card, None, None)
                            break_outside = True
                            break
                        elif isinstance(card.effect, PerBoardElement):
                            out.chains += card.chain_children
                            out.colors[card.color.lower()] += 1
                            out.guilds[card.name.lower()] = True
                            out.built[card.name] = True
                            break_outside = True
                            break
                        else:
                            out.chains += card.chain_children
                            out.colors[card.color.lower()] += 1
                            out.built[card.name] = True
                            break_outside = True
                            break

                if break_outside:
                    break
            if not break_outside:
                for card_data in card_json['guildCards']:
                    if card_data['name'] == structure:
                        card = Card.from_dict(card_data)
                        if isinstance(card.effect, PerBoardElement):
                            out.colors[card.color.lower()] += 1
                            out.guilds[card.name.lower()] = True
                            out.built[card.name] = True
                            break
                        elif isinstance(card.effect, Science):
                            out.apply_card(card, None, None)
                            break
        return out


class MoveCaller:
    def __init__(self, model):
        self.model = model
        self.reverse_card_dict = {v: k for k, v in card_dict.items()}

    @classmethod
    def prepare_obs(cls, cards, self_board, left_board, right_board, ncard, age, discarded):
        if discarded and ncard in [6, 13, 20]:
            ncard += 1
        card_array = np.zeros(80)
        np.add.at(card_array, [card_dict[card] for card in cards], 1)
        return np.concatenate([
                card_array,
                self_board.to_obs(),
                left_board.to_obs(),
                right_board.to_obs(),
                np.array([discarded]) / 20,
                np.array([(ncard - 1 + (age-1) * 7) / 21]),
                np.array([int(age == 2)])
                ])

    @classmethod
    def prepare_mask(cls, card_array, cards, own_board, left, right, ncard, age):
        player = AIPlayer()
        for card in cards:
            break_outside = False
            for age in [1, 2, 3]:
                for card_data in card_json[f'age{age}']['cards']:
                    if card_data['name'] == card:
                        player.cards.append(Card.from_dict(card_data))
                        break_outside=True
                        break
                if break_outside:
                    break
            if not break_outside:
                for card_data in card_json['guildCards']:
                    if card_data['name'] == card:
                        player.cards.append(Card.from_dict(card_data))
                        break

        player.board = own_board
        player.left = left
        player.right = right
        player.env = MockEnv((ncard - 1 + (age-1) * 7))
        return player.generate_mask(card_array)

    def __call__(self, cards, own, left, right, ncard, age, discarded):
        def remove_zero(value):
            if value == 0:
                return -999999
            else:
                return value

        obs = self.prepare_obs(cards, own, left, right, ncard, age, discarded)
        mask = self.prepare_mask(obs[:80].copy(), cards, own, left, right, ncard, age)
        model_out = self.model(obs) * mask
        iarg = np.argmax(np.vectorize(remove_zero)(model_out))
        selection = self.reverse_card_dict[iarg % 80]
        action = {0: 'build card', 1: 'build wonder', 2: 'discard'}[iarg // 80]
        return selection, action
