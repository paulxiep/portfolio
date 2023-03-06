import random
from functools import reduce

from .ai_choices import ai_choices
from .catan_player import *
from .game_parameters import *


class CatanSubAI:
    def __init__(self, personality='basis', weight=None):
        self.personality = personality
        if self.personality == 'ml':
            assert weight is not None
            self.weight = weight

    def __call__(self, session, player):
        if self.personality == 'basis':
            answer = self.basis_call(session, player)
        if self.personality == 'random':
            answer = self.random_call(session, player)
        player.last_action = answer
        return answer

    def basis_call(self, session, player):
        # to be implemented in subclasses
        pass

    @staticmethod
    def potential_settlement_weight(corner):
        return corner.pips_plus() + random.random()

    @staticmethod
    def compile_eligible_development(session, player=None):
        return session.development_pool

    @staticmethod
    def compile_ineligible_settlement(session):
        return session.compile_ineligible_settlement()
    @staticmethod
    def compile_eligible_settlement(session, player):
        return session.compile_eligible_settlement(player)

    @staticmethod
    def compile_eligible_city(session, player):
        return session.compile_eligible_city(player)

    @staticmethod
    def compile_eligible_road(session, player):
        return session.compile_eligible_road(player)

class PlentyAI(CatanSubAI):
    def basis_call(self, session, player):
        for build_option in ['city', 'settlement', 'road']:
            if len(getattr(self, f'compile_eligible_{build_option}')(session, player)) > 0:
                budget = {k: min(player.resource[k] - v, 0) for k, v in build_options[build_option].items()}
                lacking = reduce(int.__add__, list(budget.values()))
                if lacking == -2:
                    out = []
                    for k, v in budget.items():
                        if v == -2:
                            return (k, k)
                        elif v == -1:
                            out.append(k)
                    return out
                if lacking == -1:
                    for k, v in budget.items():
                        if v == -1:
                            return (k, min([(r, p) for r, p in session.player_pip(player.name).items()],
                                                     key=lambda x: x[1] + random.random())[0])
    def random_call(self, session, player):
        return (random.choice(resource_types), random.choice(resource_types))


class MonopolyAI(CatanSubAI):
    def basis_call(self, session, player):
        max_gain = max(
            [(resource, p.resource[resource]) for resource in resource_types for p in session.players.values() if p.name != player.name],
            key=lambda x: x[1])
        return max_gain[0]
    def random_call(self, session, player):
        return random.choice(resource_types)


class DevelopmentAI(CatanSubAI):
    def basis_call(self, session, player):
        if player.development['road'] > 0:
            return ['road']
        if player.development['monopoly'] > 0:
            max_gain = max(
                [(resource, p.resource[resource]) for resource in resource_types for p in session.players.values() if
                 p.name != player.name], key=lambda x: x[1])
            if max_gain[1] > len(session.players.keys()):
                return ['monopoly']
        if player.development['plenty'] > 0:
            for build_option in ['city', 'settlement', 'road']:
                if len(getattr(self, f'compile_eligible_{build_option}')(session, player)) > 0:
                    budget = {k: min(player.resource[k] - v, 0) for k, v in build_options[build_option].items()}
                    if -2 <= reduce(int.__add__, list(budget.values())) < 0:
                        return ['plenty']
        return []
    def random_call(self, session, player):
        return random.choice([[dev] * int(player.development[dev] > 0) for dev in ['road', 'plenty', 'monopoly']] + [[]])


class KnightAI(CatanSubAI):
    def basis_call(self, session, player):
        if player.development['knight'] < 1:
            return False
        else:
            for hex in session.board.hex_list:
                if hex.robber and hex.resource!='desert':
                    for corner in hex.corners.values():
                        if corner is not None and (corner.settlement==player.name or corner.city==player.name):
                            # print('\n\n\n\n\n\n\n\n\n')
                            return True
                    break
        return False
    def random_call(self, session, player):
        if player.development['knight'] < 1 or random.random()<0.5:
            return False
        else:
            return True

class RobberAI(CatanSubAI):
    def basis_call(self, session, player):
        def hex_score(hex):
            if hex.robber or hex.resource == 'desert':
                return -10000
            pips = hex.pips()
            return reduce(int.__add__, [pips * (corner.settlement is not None and corner.settlement != player) + \
                                        2 * pips * (corner.city is not None and corner.city != player) - \
                                        (100 * pips * (corner.settlement is not None and corner.settlement == player) + \
                                         200 * pips * (corner.city is not None and corner.city == player))
                                        for corner in hex.corners.values() if corner is not None])

        return max(session.board.hex_list, key=hex_score).coor
    def random_call(self, session, player):
        def hex_score(hex):
            if hex.robber or hex.resource == 'desert':
                return -10000
            pips = hex.pips()
            return reduce(float.__add__, [random.random() - \
                                        (100 * pips * (corner.settlement is not None and corner.settlement == player) + \
                                         200 * pips * (corner.city is not None and corner.city == player))
                                        for corner in hex.corners.values() if corner is not None])

        return max(session.board.hex_list, key=hex_score).coor



class RoadAI(CatanSubAI):
    def basis_call(self, session, player):
        eligible_settlements = self.compile_eligible_settlement(session, player)
        road_choices = self.compile_eligible_road(session, player)
        gainful_choices = []
        secondary = []
        for choice in road_choices:
            secondary.append(session.board.edges[choice[1]][choice[0]])

            session.board.edges[choice[1]][choice[0]].road = player
            new = list(set(self.compile_eligible_settlement(session, player)) - set(eligible_settlements))
            if len(new) > 0:
                gainful_choices.append((choice, session.board.corners[new[0][1]][new[0][0]].pips_plus() + random.random()))
            session.board.edges[choice[1]][choice[0]].road = None
        if len(gainful_choices) > 0:
            coor_choice = max(gainful_choices, key=lambda x: x[1] + random.random())[0]
        else:
            ineligible_settlements = self.compile_ineligible_settlement(session)
            try:
                coor_choice = max(secondary, key=lambda x: self.potential_settlement_weight(max(reduce(list.__add__, [
                    [corner for corner in edge.corners.values() if
                     corner is not None and corner.coor not in ineligible_settlements] for edge in x.edges.values() if
                    edge is not None and edge.road is None]), key=self.potential_settlement_weight))).coor
            except:
                print('2 step road planning failed')
                coor_choice = random.choice(road_choices)
        return coor_choice
    def random_call(self, session, player):
        return random.choice(self.compile_eligible_road(session, player))

class SettlementAI(CatanSubAI):
    def basis_call(self, session, player):
        placement_choice = max(self.compile_eligible_settlement(session, player),
                               key=lambda coor: self.potential_settlement_weight(session.board.corners[coor[1]][coor[0]]))
        return placement_choice
    def random_call(self, session, player):
        return random.choice(self.compile_eligible_settlement(session, player))


class CityAI(CatanSubAI):
    def basis_call(self, session, player):
        placement_choice = max(self.compile_eligible_city(session, player),
                               key=lambda coor: session.board.corners[coor[1]][coor[0]].pips() + random.random())
        return placement_choice
    def random_call(self, session, player):
        return random.choice(self.compile_eligible_city(session, player))


class TradeAI(CatanSubAI):
    def basis_call(self, session, player):
        def trade_weight(resource, required, trade_rates):
            amount = player.resource[resource]
            if resource in required.keys():
                amount -= required[resource]
            amount -= trade_rates[resource]
            if amount < 0:
                return -100
            else:
                out = amount + random.random() / 2 + (4 - trade_rates[resource]) / 4
                # print('weight', out)
                return out

        def try_trade(build_option):
            resource = max([resource for resource in resource_types],
                           key=lambda x: trade_weight(x, build_options['city'], trade_rates))
            if trade_weight(resource, build_options[build_option], trade_rates) > 0:
                return (resource, [k for k, v in budget.items() if v < 0][0])
            return False

        trade_rates = player.trade_rates
        for tangible in ['city', 'settlement', 'road']:
            budget = {k: player.resource[k] - v for k, v in build_options[tangible].items()}
            if len(getattr(self, f'compile_eligible_{tangible}')(session, player)) > 0:
                lacking = reduce(int.__add__, [min(v, 0) for v in budget.values()])
                # print(tangible, lacking)
                if lacking == 0:
                    return []
                elif lacking == -1:
                    trade = try_trade(tangible)
                    # print('trade', trade)
                    if trade:
                        return [trade]
        return []
    def random_call(self, session, player):
        trade_rates = player.trade_rates
        choices = [resource for resource, rate in trade_rates.items() if player.resource[resource]>=rate]
        # print(choices)
        if len(choices)>0 and random.random()>0.5:
            choice = random.choice(choices)
            # print(choice)
            return [(choice, random.choice([resource for resource in resource_types if resource!= choice]))]
        else:
            return []

class BuildAI(CatanSubAI):
    def basis_call(self, session, player):
        eligible_cities = self.compile_eligible_city(session, player)
        save_for_city = False
        if len(eligible_cities) > 0:
            required = [min(player.resource[k] - v, 0) for k, v in build_options['city'].items()]
            if reduce(int.__add__, required) == 0:
                return [('city', player.sub_ais['city_ai'](session, player))]
                # for k, v in build_options['city'].items():
                #     player.resource[k] -= v
                #     self.resource_data[player][k].set(self.players[player][k])
                # self.city_ai(eligible=eligible_cities)
                # print(f'{player} built city')
            elif reduce(int.__add__, required) >= -1:
                save_for_city = True
                # print(f'{player} saving up for city')
        eligible_settlements = self.compile_eligible_settlement(session, player)
        if len(eligible_settlements) > 0:
            required = [player.resource[k] >= v for k, v in build_options['settlement'].items()]
            if all(required):
                return [('settlement', player.sub_ais['settlement_ai'](session, player))]
                # for k, v in build_options['settlement'].items():
                #     self.players[player][k] -= v
                #     self.resource_data[player][k].set(self.players[player][k])
                # self.settlement_ai(eligible=eligible_settlements)
                # print(f'{player} built settlement')
            if reduce(lambda x, y: x + int(y), required, 0) == 3:
                # print(f'{player} saving up for settlement')
                return []
        # else:
        eligible_roads = self.compile_eligible_road(session, player)
        # print('eligible_roads', eligible_roads)
        if len(eligible_roads) > 0:
            required = [player.resource[k] >= v for k, v in build_options['road'].items()]
            if all(required):
                return [('road', player.sub_ais['road_ai'](session, player))]
        if not save_for_city and len(session.development_pool) > 0:
            required = [player.resource[k] >= v for k, v in build_options['development'].items()]
            if all(required):
                return [('development', None)]
        return []
    def random_call(self, session, player):
        choices = []
        if all([player.resource[k] >= v for k, v in build_options['city'].items()])\
            and len(self.compile_eligible_city(session, player)) > 0:
            choices.append('city')
        if all([player.resource[k] >= v for k, v in build_options['settlement'].items()])\
            and len(self.compile_eligible_settlement(session, player)) > 0:
            choices.append('settlement')
        if all([player.resource[k] >= v for k, v in build_options['road'].items()])\
            and len(self.compile_eligible_road(session, player)) > 0:
            choices.append('road')
        if all([player.resource[k] >= v for k, v in build_options['development'].items()])\
            and len(self.compile_eligible_development(session, player)) > 0:
            choices.append('development')
        if len(choices)>0 and random.random() > 0.3:
            # print(choices)
            choice = random.choice(choices)
            # print(choice, getattr(self, f'compile_eligible_{choice}')(session, player))
            if choice != 'development':
                return [(choice, random.choice(getattr(self, f'compile_eligible_{choice}')(session, player)))]
            else:
                return [(choice, None)]
        else:
            return []

class SetUpAI(CatanSubAI):
    @staticmethod
    def compile_eligible_road(session, player):
        return session.compile_eligible_road_initial(player)

    @staticmethod
    def compile_eligible_settlement(session, player):
        return session.compile_eligible_settlement_initial()


class FirstSettlementAI(SetUpAI, SettlementAI):
    pass


class SecondSettlementAI(SetUpAI, SettlementAI):
    pass


class InitialRoadAI(SetUpAI, RoadAI):
    pass

def from_camel(phrase):
    def title_ai(word):
        if word == 'ai':
            return 'AI'
        else:
            return word.title()
    return ''.join(list(map(title_ai, phrase.split('_'))))

sub_ais = {sub_ai: {choice: globals()[from_camel(sub_ai)](choice, None)
                    for choice in choices}
           for sub_ai, choices in ai_choices.items()}

class CatanAI(CatanPlayer):
    def __init__(self, personality=None, *args):
        super().__init__(*args)
        self.last_action = None
        if isinstance(personality, str):
            personalities = {sub_ai: personality for sub_ai in sub_ais.keys()}
        else:
            personalities = {sub_ai: random.choice(list(choices.keys())) for sub_ai, choices in sub_ais.items()}
            if isinstance(personality, dict):
                personalities.update(personality)
        self.personalities = personalities
        self.sub_ais = {sub_ai: sub_ais[sub_ai][personality] for sub_ai, personality in self.personalities.items()}
