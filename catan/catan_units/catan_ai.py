import itertools

from utils.compiled_parameters import *

from .ai_choices import ai_choices
from .ai_utils import *
from .catan_player import *
from .game_parameters import *


class CatanAI(CatanPlayer):
    def __init__(self, personality=None, *args, scalings={}):
        '''
        :param personality:
            if None, will be randomized from available personalities
            if str, will assign same personality to all sub AIs
            if dict, will assign specified personalities to specified sub AIs, the rest will be random
        :param scalings:
            this will be used from the 3rd development phase onwards
        '''
        super().__init__(*args)
        self.last_action = None
        if isinstance(personality, str):
            personalities = {sub_ai: personality for sub_ai in ai_choices.keys()}
        else:
            personalities = {sub_ai: random.choice(choices) for sub_ai, choices in ai_choices.items()}
            if isinstance(personality, dict):
                personalities.update(personality)
        self.personalities = personalities
        self.sub_ais = {sub_ai: globals()[from_camel(sub_ai)](personality, scalings=scalings) for sub_ai, personality in
                        self.personalities.items()}
        self.priorities = []
        self.strategy = None


class CatanSubAI:
    '''
    common base class of all sub AIs
    '''

    def __init__(self, personality='basis', weight=None, scalings={}):
        '''
        :param personality:
            currently supports 'basis' and 'random'.
            'basis_v2' should be added in the 3rd development stage
        :param weight: will be used when any ML is needed
        :param scalings: will be used from the 3rd development stage onwards
        '''
        self.personality = personality
        self.pips_scaling = 1
        self.random_scaling = scalings.get('random', 1 / 4)
        self.diversity_scaling = scalings.get('diversity', 4)
        self.distance_scaling = scalings.get('distance', 1 / 6)
        self.generic_harbor_scaling = scalings.get('generic_harbor', 1 / 6)
        self.resource_harbor_scaling = scalings.get('resource_harbor', 1 / 4)

    def __call__(self, session, player, *args):
        '''
        so the class can be called like a function
        '''
        answer = getattr(self, f'{self.personality}_call')(session, player, *args)
        player.last_action = answer
        return answer

    def make_state_data(self, session, player):
        '''
        this is a relic from 2nd stage's pure ML AI experiments,
        but may become useful again in the future
        '''

        def pips(number):
            return (6 - abs(7 - number)) * (number != 0)

        def transform_corner(settlement, city):
            if settlement != 'none':
                out1 = settlement
                out2 = 1
            elif city != 'none':
                out1 = city
                out2 = 2
            else:
                out1 = 'none'
                out2 = 0
            return out1, out2

        static_data = session.harbors + session.resources + list(map(pips, session.numbers))
        corners = reduce(list.__add__, [[corner.settlement, corner.city] for corner in session.board.corner_list])
        edges = [edge.road for edge in session.board.edge_list]
        hexes = [[int(hex.robber) for hex in session.board.hex_list].index(1)]
        player_list = list(session.players.keys())
        player_index = player_list.index(player.name)
        player_dict = {player.name: 'self', player_list[player_index - 3]: 'left',
                       player_list[player_index - 2]: 'across', player_list[player_index - 1]: 'right'}
        corner_data = list(map(lambda x: 'none' if x is None else player_dict[x], corners))
        for i in range(len(corner_data) // 2):
            temp1, temp2 = transform_corner(corner_data[2 * i], corner_data[2 * i + 1])
            corner_data[2 * i] = temp1
            corner_data[2 * i + 1] = temp2
        edge_data = list(map(lambda x: 'none' if x is None else player_dict[x], edges))
        return static_data + corner_data + edge_data + hexes + reduce(list.__add__,
                                                                      [list(session.players[value].resource.values()) + \
                                                                       list(session.players[
                                                                                value].development.values()) + \
                                                                       list(session.players[value].tokens.values()) + \
                                                                       list(session.players[value].trade_rates.values())
                                                                       for value in player_dict.keys()])

    def basis_call(self, session, player):
        # to be implemented in subclasses
        pass

    def settlement_weight(self, factors, corner=None, player_pips=None):
        '''
        Used to compute weight for placing potential settlements (and cities too)
        :param factors: factors to consider as a list. Consult ai_utils.py for options
        :return:
        '''
        weight = 0
        for factor in factors:
            weight += globals()[f'{factor}_weight'](corner, player_pips) * getattr(self, f'{factor}_scaling')
        return weight

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
                            return (k, min([(r, p) for r, p in session.player_pips(player).items()],
                                           key=lambda x: x[1] + random.random())[0])

    def random_call(self, session, player):
        return (random.choice(resource_types), random.choice(resource_types))


class MonopolyAI(CatanSubAI):
    def basis_call(self, session, player):
        max_gain = max(
            [(resource, p.resource[resource]) for resource in resource_types for p in session.players.values() if
             p.name != player.name],
            key=lambda x: x[1])
        return max_gain[0]

    def random_call(self, session, player):
        return random.choice(resource_types)


class DevelopmentAI(CatanSubAI):
    def basis_call(self, session, player):
        if player.development['road'] > 0 and len(self.compile_eligible_road(session, player)) > 1 and player.tokens[
            'road'] > 1:
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
        return random.choice(
            [[dev] * int(player.development[dev] > 0) for dev in ['road', 'plenty', 'monopoly']] + [[]])


class KnightAI(CatanSubAI):
    def basis_call(self, session, player):
        if player.development['knight'] < 1:
            return False
        else:
            for hex in session.board.hex_list:
                if hex.robber and hex.resource != 'desert':
                    for corner in hex.corners.values():
                        if corner is not None and (corner.settlement == player.name or corner.city == player.name):
                            return True
                    break
        return False

    def random_call(self, session, player):
        if player.development['knight'] < 1 or random.random() < 0.5:
            return False
        else:
            return True


class RobberAI(CatanSubAI):
    def basis_call(self, session, player):
        def hex_score(hex):
            if hex.robber or hex.resource == 'desert':
                return -10000
            pips = hex.pips() + random.random()
            return reduce(float.__add__, [pips * (corner.settlement is not None and corner.settlement != player.name) + \
                                          2 * pips * (corner.city is not None and corner.city != player.name) - \
                                          (100 * pips * (
                                                  corner.settlement is not None and corner.settlement == player.name) + \
                                           200 * pips * (corner.city is not None and corner.city == player.name))
                                          for corner in hex.corners.values() if corner is not None])

        return max(session.board.hex_list, key=hex_score).coor

    def random_call(self, session, player):
        def hex_score(hex):
            if hex.robber or hex.resource == 'desert':
                return -10000
            pips = hex.pips()
            return reduce(float.__add__, [random.random() - \
                                          (100 * pips * (
                                                  corner.settlement is not None and corner.settlement == player.name) + \
                                           200 * pips * (corner.city is not None and corner.city == player.name))
                                          for corner in hex.corners.values() if corner is not None])

        return max(session.board.hex_list, key=hex_score).coor


class RoadAI(CatanSubAI):
    def basis_call(self, session, player):
        eligible_settlements = self.compile_eligible_settlement(session, player)
        road_choices = self.compile_eligible_road(session, player)
        gainful_choices = []
        secondary = []
        player_pips = session.player_pips(player)
        for choice in road_choices:
            secondary.append(session.board.edges[choice[1]][choice[0]])

            session.board.edges[choice[1]][choice[0]].road = player
            new = list(set(self.compile_eligible_settlement(session, player)) - set(eligible_settlements))
            if len(new) > 0:
                gainful_choices.append((choice, self.settlement_weight(
                    ['pips', 'generic_harbor', 'resource_harbor', 'diversity', 'random'],
                    session.board.corners[new[0][1]][new[0][0]], player_pips)))
            session.board.edges[choice[1]][choice[0]].road = None
        if len(gainful_choices) > 0:
            coor_choice = max(gainful_choices, key=lambda x: x[1] + random.random())[0]
        else:
            ineligible_settlements = self.compile_ineligible_settlement(session)
            try:
                coor_choice = max(secondary,
                                  key=lambda x: self.settlement_weight(
                                      ['pips', 'generic_harbor', 'resource_harbor', 'diversity', 'random'],
                                      max(reduce(list.__add__, [
                                          [corner for corner in edge.corners.values() if
                                           corner is not None and corner.coor not in ineligible_settlements] for edge in
                                          x.edges.values() if
                                          edge is not None and edge.road is None]),
                                          key=lambda z: self.settlement_weight(
                                              ['pips', 'generic_harbor', 'resource_harbor', 'diversity', 'random'], z,
                                              player_pips)), player_pips)).coor
            except:
                print('2 step road planning failed')
                coor_choice = random.choice(road_choices)
        return coor_choice

    def random_call(self, session, player):
        return random.choice(self.compile_eligible_road(session, player))


class SettlementAI(CatanSubAI):
    def basis_call(self, session, player):
        player_pips = session.player_pips(player)
        placement_choice = max(self.compile_eligible_settlement(session, player),
                               key=lambda coor: self.settlement_weight(
                                   ['pips', 'generic_harbor', 'resource_harbor', 'diversity', 'random'],
                                   session.board.corners[coor[1]][coor[0]], player_pips))
        return placement_choice

    def random_call(self, session, player):
        return random.choice(self.compile_eligible_settlement(session, player))


class CityAI(CatanSubAI):
    def basis_call(self, session, player):
        placement_choice = max(self.compile_eligible_city(session, player),
                               key=lambda coor: self.settlement_weight(['pips', 'random'],
                                                                       session.board.corners[coor[1]][coor[0]]))
        return placement_choice

    def random_call(self, session, player):
        return random.choice(self.compile_eligible_city(session, player))


class PlayerTradeAI(CatanSubAI):
    def basis_call(self, session, player, trade_with=None):
        i = 0
        required = []
        for priority in player.priorities:
            if i < 3:
                i += 1
                required.append(build_options[priority])
        hand = {resource: player.resource[resource] for resource in resource_types}
        count = reduce(int.__add__, hand.values())
        i = 0
        plan = 0
        while i < 3 and len(required) > i:
            if count >= reduce(int.__add__, [reduce(int.__add__, required[j].values()) for j in range(i + 1)]):
                plan += 1
                i += 1
            else:
                break
        all_required = reduce(lambda x, y: {resource: x[resource] + y.get(resource, 0) for resource in resource_types},
                              required[:plan], {resource: 0 for resource in resource_types})
        expendables = {resource: hand[resource] - all_required[resource] for resource in resource_types if
                       hand[resource] - all_required[resource] > 0}
        lacking = {resource: all_required[resource] - hand[resource] for resource in resource_types if
                   all_required[resource] - hand[resource] > 0}
        if trade_with is not None:
            potential = {resource: min(trade_with.resource[resource], lacking[resource]) for resource in lacking.keys()
                         if min(trade_with.resource[resource], lacking[resource]) > 0}
            options = list(map(lambda x: f'{player.name}: {x[0]} for {x[1]}',
                               list(itertools.product(potential.keys(), expendables.keys()))))

            return options
        else:
            trades = []
            random.shuffle(session.trade_list)
            for trade in session.trade_list:
                trade_with, trading = trade.split(': ')
                trade_away, trade_for = trading.split(' for ')
                if trade_away in expendables.keys() and trade_for in lacking.keys():
                    expendables[trade_away] -= 1
                    if expendables[trade_away] == 0:
                        expendables.pop(trade_away)
                    lacking[trade_for] -= 1
                    if lacking[trade_for] == 0:
                        lacking.pop(trade_for)
                    trades.append((trade_with, trade_away, trade_for))
            return trades


class BankTradeAI(CatanSubAI):
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
                if lacking == 0:
                    return []
                elif lacking == -1:
                    trade = try_trade(tangible)
                    if trade:
                        return [trade]
        return []

    def random_call(self, session, player):
        trade_rates = player.trade_rates
        choices = [resource for resource, rate in trade_rates.items() if player.resource[resource] >= rate]
        spare_choices = [resource for resource, rate in trade_rates.items() if player.resource[resource] >= rate + 1]
        high_spare_choices = [resource for resource, rate in trade_rates.items() if
                              player.resource[resource] >= rate + 2]
        super_spare_choices = [resource for resource, rate in trade_rates.items() if
                               player.resource[resource] >= rate + 3]
        choices = [resource for resource in choices if resource not in spare_choices]
        spare_choices = [resource for resource in spare_choices if resource not in high_spare_choices]
        high_spare_choices = [resource for resource in high_spare_choices if resource not in super_spare_choices]

        if len(super_spare_choices) > 0 and random.random() > 0.6:
            choice = random.choice(super_spare_choices)
            return [(choice, random.choice([resource for resource in resource_types if resource != choice]))]
        elif len(high_spare_choices) > 0 and random.random() > 0.5:
            choice = random.choice(high_spare_choices)
            return [(choice, random.choice([resource for resource in resource_types if resource != choice]))]
        elif len(spare_choices) > 0 and random.random() > 0.4:
            choice = random.choice(spare_choices)
            return [(choice, random.choice([resource for resource in resource_types if resource != choice]))]
        elif len(choices) > 0 and random.random() > 0.3:
            choice = random.choice(choices)
            return [(choice, random.choice([resource for resource in resource_types if resource != choice]))]
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
            elif reduce(int.__add__, required) >= -1:
                save_for_city = True
        eligible_settlements = self.compile_eligible_settlement(session, player)
        if len(eligible_settlements) > 0:
            required = [player.resource[k] >= v for k, v in build_options['settlement'].items()]
            if all(required):
                return [('settlement', player.sub_ais['settlement_ai'](session, player))]
            if reduce(lambda x, y: x + int(y), required, 0) == 3:
                return []

        eligible_roads = self.compile_eligible_road(session, player)
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
        if all([player.resource[k] >= v for k, v in build_options['city'].items()]) \
                and len(self.compile_eligible_city(session, player)) > 0:
            choices.append('city')
        if all([player.resource[k] >= v for k, v in build_options['settlement'].items()]) \
                and len(self.compile_eligible_settlement(session, player)) > 0:
            choices.append('settlement')
        if all([player.resource[k] >= v for k, v in build_options['road'].items()]) \
                and len(self.compile_eligible_road(session, player)) > 0:
            choices.append('road')
        if all([player.resource[k] >= v for k, v in build_options['development'].items()]) \
                and len(self.compile_eligible_development(session, player)) > 0:
            choices.append('development')
        if len(choices) > 0 and random.random() > 0.3:
            choice = random.choice(choices)
            if choice != 'development':
                return [(choice, random.choice(getattr(self, f'compile_eligible_{choice}')(session, player)))]
            else:
                return [(choice, None)]
        else:
            return []


class StrategyAI(CatanSubAI):
    def basis_call(self, session, player):
        priorities = []
        player_pips = session.player_pips(player)
        active_player_pips = session.active_player_pips(player)
        if reduce(int.__add__, player_pips.values(), 0) - \
                reduce(int.__add__, active_player_pips.values(), 0) > 4:
            print('robber on hex, should buy development')
            priorities += ['development']
        eligible_cities = self.compile_eligible_city(session, player)
        if len(self.compile_eligible_settlement(session, player)) < 1:
            if len(self.compile_eligible_road(session, player)) < 1:
                if len(eligible_cities) > 0:
                    priorities += ['city']
            else:
                priorities += ['road']
        else:
            priorities += ['settlement']
        if 'development' not in priorities:
            priorities += ['development']
        if 'city' not in priorities:
            priorities += ['city']
        print(player.name, 'priorities', priorities)
        player.priorities = priorities


class SetUpAI(CatanSubAI):
    '''
    the setup stage will utilize road AI and settlement AI,
    but the optimal strategy is likely different from main session strategy.
    So the setup AI exists as a subclass to override main session's AIs
    The eligible roads and settlements are also implemented differently
    '''
    @staticmethod
    def compile_eligible_road(session, player):
        return session.compile_eligible_road_initial(player)

    @staticmethod
    def compile_eligible_settlement(session, player):
        return session.compile_eligible_settlement_initial()


class FirstSettlementAI(SetUpAI, SettlementAI):
    def basis_call(self, session, player):
        '''
        1st settlement is to be selected based on raw pips count
        '''
        placement_choice = max(self.compile_eligible_settlement(session, player),
                               key=lambda coor: self.settlement_weight(['pips', 'random'],
                                                                       session.board.corners[coor[1]][coor[0]]))
        return placement_choice


class SecondSettlementAI(SetUpAI, SettlementAI):
    def basis_call(self, session, player):
        '''
        difference from main session AI is the inclusion of distance factor
        '''
        player_pips = session.player_pips(player)
        placement_choice = max(self.compile_eligible_settlement(session, player),
                               key=lambda coor: self.settlement_weight(
                                   ['pips', 'generic_harbor', 'resource_harbor', 'diversity', 'random'],
                                   session.board.corners[coor[1]][coor[0]], player_pips) \
                                                + session.board.corners[player.settlements[0][1]][
                                                    player.settlements[0][0]].distance(
                                   session.board.corners[coor[1]][coor[0]]) * self.distance_scaling)
        return placement_choice


class InitialRoadAI(SetUpAI, RoadAI):
    '''
    still needed even if nothing is defined. The eligible roads are different for the setup stage.
    '''
    pass



