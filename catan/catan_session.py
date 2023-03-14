from catan_units.catan_hex import *
from catan_units.catan_player import *
from catan_units.game_parameters import *
from catan_units.catan_ai import CatanAI
import random
import logging


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(logging.Formatter('action: %(message)s'))

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


first_actions = ['settlement', 'road']
second_actions = ['settlement', 'collect_resource', 'road']
turn_actions = ['dice', 'resolve_dice', 'trade', 'build']


class CatanSession:
    def __init__(self, players, board_type=4, board=None, actionlogfile='action_log.txt'):
        self.board_type = board_type
        if board is not None:
            self.board = board
        else:
            self.board = CatanHexBoard(board_type)
        harbor_list = [x[0] for x in [((1, 2), (1, 1)), ((2, 1), (3, 2)), ((4, 3), (5, 4)),
                                      ((6, 6), (6, 7)), ((5, 9), (4, 10)), ((3, 11), (2, 12)),
                                      ((1, 12), (1, 11)), ((1, 9), (1, 8)), ((1, 5), (1, 4))]]
        self.players = {player: self.generate_player(personality, player, color, ai) for player, color, ai, personality
                        in players}
        self.development_pool = development_cards.copy()
        self.game_over = False
        self.game_state_generator = self.game_states()
        self.game_state = []
        self.board.assign_resources()
        self.board.assign_numbers()

        self.harbors = [corner.harbor for corner in self.board.corner_list if corner.coor in harbor_list]
        self.resources = [hex.resource for hex in self.board.hex_list]
        self.numbers = [hex.number for hex in self.board.hex_list]

        self.dice = 0
        self.pending_process = []
        self.plenty = False
        self.monopoly = False
        self.road = False
        self.action_logger = setup_logger(actionlogfile, actionlogfile)

    def generate_player(self, personality, player, color, ai):
        if ai:
            return CatanAI(personality, player, color)
        else:
            return CatanPlayer(player, color)

    def instruction(self, instruction):
        '''
        for logging actions taken in a simulation game
        '''
        self.action_logger.info(instruction)

    def game_states(self):
        '''
        generator to generate game states
        '''
        for player in self.players.keys():
            for action in first_actions:
                yield ('setup_1', player, action)
        for player in reversed(self.players.keys()):
            for action in second_actions:
                yield ('setup_2', player, action)
        turn = 1
        while not self.game_over:
            for player in self.players.keys():
                for action in turn_actions:
                    yield (f'turn_{turn}', player, action)
            turn += 1

    def compile_eligible_development(self, player=None):
        return self.development_pool

    def compile_ineligible_settlement(self):
        ineligible = []
        for j, row in self.board.corners.items():
            for i, corner in row.items():
                if self.board.corners[j][i].settlement is not None or self.board.corners[j][i].city is not None:
                    ineligible.append((i, j))
                    ineligible.append((i, j - 1))
                    ineligible.append((i, j + 1))
                    if self.board.corners[j][i].type == 'top':
                        if j in range(1, self.board.d_length * 2 + 1):
                            ineligible.append((i + 1, j + 1))
                        else:
                            ineligible.append((i - 1, j + 1))
                    else:
                        if j in range(1, self.board.d_length * 2 + 1):
                            ineligible.append((i - 1, j - 1))
                        else:
                            ineligible.append((i + 1, j - 1))
        return ineligible

    def compile_eligible_settlement(self, player):
        if player.tokens['settlement'] == 0:
            return []
        eligible = []
        for edge in self.board.edge_list:
            eligible += [corner.coor for corner in edge.corners.values() if edge.road == player.name]
        return [coor for coor in eligible if coor not in self.compile_ineligible_settlement()]

    def compile_eligible_city(self, player):
        if player.tokens['city'] == 0:
            return []
        eligible = []
        for corner in self.board.corner_list:
            if corner.settlement == player.name:
                eligible += [corner.coor]
        return eligible

    def compile_eligible_road(self, player):
        if player.tokens['road'] == 0:
            return []
        eligible = []
        for corner in self.board.corner_list:
            if corner.settlement == player.name or corner.city == player.name:
                eligible += [edge.coor for edge in corner.edges.values() if edge is not None]
        for edge in self.board.edge_list:
            if edge.road == player.name:
                eligible += reduce(list.__add__,
                                   [[edge2.coor for edge2 in corner.edges.values() if edge2 is not None] for corner in
                                    edge.corners.values() if corner is not None and ((
                                                                                             corner.settlement is None and corner.city is None) or corner.settlement == player or corner.city == player)])
        return list(set(filter(lambda x: self.board.edges[x[1]][x[0]].road is None, eligible)))

    def compile_eligible_road_initial(self, player):
        return [edge.coor for edge in self.board.corners[player.last_action[1]][player.last_action[0]].edges.values() if
                edge is not None and edge.road is None]

    def compile_eligible_settlement_initial(self):
        return [corner.coor for corner in self.board.corner_list if
                corner.coor not in self.compile_ineligible_settlement()]

    def bank_trade(self, player, trade_away=None, trade_for=None):
        if trade_away is None and trade_for is None:
            if self.trade_away_var.get() == '' or self.trade_for_var.get() == '':
                print('trade option empty')
                self.instruction_label.config(text=f'trade option empty')
                return False
            trade_away = self.trade_away_var.get()
            trade_for = self.trade_for_var.get()
        if self.players[player].resource[trade_away] < self.players[player].trade_rates[trade_away]:
            self.instruction(f'not enough {trade_away} for trade')
            return False
        else:
            self.players[player].resource[trade_away] -= self.players[player].trade_rates[trade_away]
            self.players[player].resource[trade_for] += 1
            self.instruction(
                f'{player} exchanged {self.players[player].trade_rates[trade_away]} {trade_away} for 1 {trade_for}')
            return True

    def use_plenty(self, player):
        temp = self.players[player].sub_ais['plenty_ai'](self, self.players[player])
        print(temp)
        try:
            resource1, resource2 = temp
        except:
            resource1 = random.choice(resource_types)
            resource2 = random.choice(resource_types)
        self.players[player].resource[resource1] += 1
        self.instruction(f'{player} chose {resource1} for plenty')
        # resource = self.players[player].sub_ais['plenty_ai']
        self.players[player].resource[resource2] += 1
        self.instruction(f'{player} chose {resource2} for plenty')
        self.plenty = False

    def use_monopoly(self, current_player):
        resource = self.players[current_player].sub_ais['monopoly_ai'](self, self.players[current_player])
        take_text = []
        for player in self.players.keys():
            if player != current_player:
                resource_count = self.players[player].resource[resource]
                if resource_count > 0:
                    self.players[player].resource[resource] = 0
                    self.players[current_player].resource[resource] += resource_count
                    take_text += [f'{current_player} takes {resource_count} {resource} from {player}']
        self.instruction(',\n'.join(take_text))
        self.monopoly = False

    def use_road(self, player):
        '''
        this dummy function is required so its override in CatanApp can update UI on development card data
        '''
        pass

    def check_largest_army(self, player):
        if self.players[player].army > 2 and not self.players[player].largest_army:
            for p in self.players.values():
                if p.name != player and p.largest_army:
                    if self.players[player].army > p.army:
                        p.largest_army = False
                        self.players[player].largest_army = True
                        self.players[player].points += 2
                        self.players[p].points -= 2
                        return None
            # no one has largest army
            self.players[player].largest_army = True
            self.players[player].points += 2

    def use_development(self, player, development=None):
        def use_knight():
            self.players[player].army += 1
            self.check_largest_army(player)
            self.moving_robber(player)

        def use_plenty():
            self.plenty = True
            self.use_plenty(player)

        def use_monopoly():
            self.monopoly = True
            self.use_monopoly(player)

        def use_road():
            self.use_road(player)
            self.buy('road', player, free=True)
            if isinstance(self.players[player], CatanAI):
                coor = self.players[player].sub_ais['road_ai'](self, self.players[player])
                self.instruction(f'{player} placed free road at {coor}')
                self.place('road')(player, coor)
                self.buy('road', player, free=True)
                coor = self.players[player].sub_ais['road_ai'](self, self.players[player])
                self.instruction(f'{player} placed free road at {coor}')
                self.place('road')(player, coor)
            else:
                self.road = True

        if development is None:
            development = self.development_var.get()
            if development == '':
                print(f"choose development card to use")
                self.instruction_label.config(text=f"choose development card to use")
                return False
        if self.players[player].development[development] < 1:
            print(f'no {development} card to use')
            self.instruction_label.config(text=f'no {development} card to use')
            return None
        else:
            self.players[player].development[development] -= 1
            self.instruction(f'{player} used {development}')
            locals()[f'use_{development}']()

    def moving_robber(self, player):
        self.pending_process.append('robber')
        if isinstance(self.players[player], CatanAI):
            self.robber_move(player, self.players[player].sub_ais['robber_ai'](self, self.players[player]),
                             self.board.robber)

    def buy(self, to_buy, player, free=False):
        if not isinstance(player, CatanAI):
            if not getattr(self, f'compile_eligible_{to_buy}')(self.players[player]):
                print(f'no eligible {to_buy}')
                return False
        if not free:
            for resource, required in build_options[to_buy].items():
                if self.players[player].resource[resource] < required:
                    print(f'not enough resource for {to_buy}')
                    return False
            for resource, required in build_options[to_buy].items():
                self.players[player].resource[resource] -= required
        if to_buy == 'development':
            self.players[player].inactive_development.append(
                self.development_pool.pop(random.randrange(len(self.development_pool))))
            if isinstance(self.players[player], CatanAI):
                self.pending_process.append('place')
        else:
            self.pending_process.append('place')
        return True

    def place(self, to_place):
        def road_place(player, coor):
            i, j = coor
            self.board.edges[j][i].road = player
            self.players[player].tokens['road'] -= 1
            self.players[player].roads.append(coor)

        def settlement_place(player, coor):
            i, j = coor
            assert self.board.corners[j][i].settlement is None
            self.board.corners[j][i].settlement = player
            self.players[player].tokens['settlement'] -= 1
            if self.board.corners[j][i].harbor is not None:
                if self.board.corners[j][i].harbor == 'x':
                    for resource in resource_types:
                        self.players[player].trade_rates[resource] = min(self.players[player].trade_rates[resource], 3)
                else:
                    self.players[player].trade_rates[self.board.corners[j][i].harbor] = 2
            self.players[player].points += 1
            self.players[player].settlements.append(coor)

        def city_place(player, coor):
            i, j = coor
            assert self.board.corners[j][i].settlement == player
            self.board.corners[j][i].settlement = None
            self.board.corners[j][i].city = player
            self.players[player].tokens['settlement'] += 1
            self.players[player].tokens['city'] -= 1
            self.players[player].points += 1
            self.players[player].settlements.remove(coor)
            self.players[player].cities.append(coor)

        def development_place(player, coor):
            # dummy function, used by AI
            pass

        self.pending_process.pop(0)
        return locals()[f'{to_place}_place']

    def robber_move(self, player, coor, robber_coor):
        print(player, coor, robber_coor)

        def has_settlement(i, j):
            out = []
            for corner in self.board.hexes[j][i].corners.values():
                if corner is not None:
                    if corner.settlement is not None:
                        out.append(corner.settlement)
                    elif corner.city is not None:
                        out.append(corner.city)
            return list(set(out))

        i, j = coor
        a, b = robber_coor
        self.board.hexes[j][i].robber = True
        self.board.hexes[b][a].robber = False
        self.board.robber = (i, j)
        self.instruction(f'{player} moved robber to {coor}')
        steal_choices = reduce(list.__add__, [[(p, resource)] * self.players[p].resource[resource] for p in
                                              filter(lambda x: x != player, has_settlement(i, j)) for resource in
                                              resource_types], [])
        to_steal = random.choice(steal_choices) if len(steal_choices) > 0 else None
        if to_steal is not None:
            self.players[to_steal[0]].resource[to_steal[1]] -= 1
            self.players[player].resource[to_steal[1]] += 1
            self.instruction(f'{player} steals 1 {to_steal[1]} from {to_steal[0]}')
        return to_steal

    def player_pips(self, player):
        out = {resource: 0 for resource in ['ore', 'brick', 'wool', 'grain', 'lumber']}
        for corner in self.board.corner_list:
            if corner.settlement == player.name:
                for resource, pip in corner.resource_pips().items():
                    out[resource] += pip
            elif corner.city == player.name:
                for resource, pip in corner.resource_pips().items():
                    out[resource] += 2 * pip
        return out

    def active_player_pips(self, player):
        out = {resource: 0 for resource in ['ore', 'brick', 'wool', 'grain', 'lumber']}
        for corner in self.board.corner_list:
            if corner.settlement == player.name:
                for resource, pip in corner.active_resource_pips().items():
                    out[resource] += pip
            elif corner.city == player.name:
                for resource, pip in corner.active_resource_pips().items():
                    out[resource] += 2 * pip
        return out
