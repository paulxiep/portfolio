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

# action_logger = setup_logger('action_logger', 'log.txt')

first_actions = ['settlement', 'road']
second_actions = ['settlement', 'collect_resource', 'road']
turn_actions = ['dice', 'resolve_dice', 'trade', 'build']

class CatanSession:
    def __init__(self, players, board_type=4, board=None, actionlogfile='action_log.txt'):
        if board is not None:
            self.board=board
        else:
            self.board = CatanHexBoard(board_type)
        self.players = {**{player: CatanPlayer(player, color) for player, color, ai in players if not ai},
                        **{player: CatanAI(None, player, color) for player, color, ai in players if ai}}
        self.development_pool = development_cards.copy()
        self.game_over = False
        self.game_state_generator = self.game_states()
        self.game_state = []
        self.board.assign_resources()
        self.board.assign_numbers()
        self.dice = 0
        self.pending_process = []
        self.plenty = False
        self.monopoly = False
        self.action_logger = setup_logger(actionlogfile, actionlogfile)

    def instruction(self, instruction):
        self.action_logger.info(instruction)

    def game_states(self):
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
        # print([self.board.corners[j][i] for j, i in ineligible if j in self.board.corners.keys() and i in self.board.corners[j].keys()])
        # return [(i, j) for j, i in ineligible if j in self.board.corners.keys() and i in self.board.corners[j].keys()]

    def compile_eligible_settlement(self, player):
        if player.tokens['settlement'] == 0:
            return []
        eligible = []
        for edge in self.board.edge_list:
            # eligible += [corner.coor for corner in edge.corners.values() if edge.road == player]
            eligible += [corner.coor for corner in edge.corners.values() if edge.road == player.name]
        # print('eligible_settlements', eligible)
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
        # eligible = [edge.coor for edge in eligible if edge is not None]
        # print('eligible_roads', eligible)
        # logging.info('compiled eligible roads')
        # logging.info(list(set(filter(lambda x: self.board.edges[x[1]][x[0]].road is None, eligible))))
        return list(set(filter(lambda x: self.board.edges[x[1]][x[0]].road is None, eligible)))


    def compile_eligible_road_initial(self, player):
        print(self.board.corners[player.last_action[1]][player.last_action[0]].edges)
        return [edge.coor for edge in self.board.corners[player.last_action[1]][player.last_action[0]].edges.values() if edge is not None and edge.road is None]

    def compile_eligible_settlement_initial(self):
        return [corner.coor for corner in self.board.corner_list if
                corner.coor not in self.compile_ineligible_settlement()]

    def bank_trade(self, player, trade_away=None, trade_for=None):
        print(player, trade_away, trade_for)
        if trade_away is None and trade_for is None:
            if self.trade_away_var.get() == '' or self.trade_for_var.get() == '':
                print('trade option empty')
                self.instruction_label.config(text=f'trade option empty')
                return False
            trade_away = self.trade_away_var.get()
            trade_for = self.trade_for_var.get()
        if self.players[player].resource[trade_away] < self.players[player].trade_rates[trade_away]:
            # print(f'not enough {trade_away} for trade')
            # print(self.players[player].resource[trade_away], self.players[player].trade_rates[trade_away])
            self.instruction(f'not enough {trade_away} for trade')
            return False
        else:
            self.players[player].resource[trade_away] -= self.players[player].trade_rates[trade_away]
            self.players[player].resource[trade_for] += 1
            # self.instruction_label.config(text=f'exchanged {trade_away} for {trade_for}')
            self.instruction(f'{player} exchanged {self.players[player].trade_rates[trade_away]} {trade_away} for 1 {trade_for}')
            return True
            # self.resource_data[player][trade_away].set(self.players[player][trade_away])
            # self.resource_data[player][trade_for].set(self.players[player][trade_for])

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
        self.instruction(take_text)
        self.monopoly = False

    def check_largest_army(self, player):
        pass

    def use_development(self, player, development=None):
        def use_knight():
            # self.instruction(f'{player} activated knight')
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
            # self.pending_process.append('road')
            try:
                self.buy('road', player, free=True)
                if isinstance(self.players[player], CatanAI):
                    coor = self.players[player].sub_ais['road_ai'](self, self.players[player])
                    self.instruction(f'{player} placed free road at {coor}')
                    self.place('road')(player, coor)
                    self.buy('road', player, free=True)
                    coor = self.players[player].sub_ais['road_ai'](self, self.players[player])
                    self.instruction(f'{player} placed free road at {coor}')
                    self.place('road')(player, coor)
            except:
                pass
        if development is None:
            development = self.development_var.get()
            if development == '':
                print(f"choose development card to use")
                self.instruction_label.config(text=f"choose development card to use")
                return False
        # print(player, development)
        if self.players[player].development[development] < 1:
            print(f'no {development} card to use')
            self.instruction_label.config(text=f'no {development} card to use')
            return None
        else:
            self.players[player].development[development] -= 1
            self.instruction(f'{player} used {development}')
            locals()[f'use_{development}']()

    def moving_robber(self, player):
        # self.instruction(f'{player} move robber')
        self.pending_process.append('robber')
        if isinstance(self.players[player], CatanAI):
            self.robber_move(player, self.players[player].sub_ais['robber_ai'](self, self.players[player]), self.board.robber)


    def buy(self, to_buy, player, free=False):
        for resource, required in build_options[to_buy].items():
            if self.players[player].resource[resource] < required:
                print(f'not enough resource for {to_buy}')
                return False
        if not isinstance(player, CatanAI):
            if not getattr(self, f'compile_eligible_{to_buy}')(self.players[player]):
                print(f'no eligible {to_buy}')
                return False
        if not free:
            for resource, required in build_options[to_buy].items():
                self.players[player].resource[resource] -= required
        if to_buy == 'development':
            self.players[player].inactive_development.append(
                self.development_pool.pop(random.randrange(len(self.development_pool))))
        if isinstance(self.players[player], CatanAI):
            self.pending_process.append('place')
        print('should return True')
        return True

    def place(self, to_place):
        def road_place(player, coor):
            print(player, coor)
            i, j = coor
            self.board.edges[j][i].road=player
            self.players[player].tokens['road'] -= 1

        def settlement_place(player, coor):
            print(player, coor)
            i, j = coor
            assert self.board.corners[j][i].settlement is None
            self.board.corners[j][i].settlement=player
            self.players[player].tokens['settlement'] -= 1
            if self.board.corners[j][i].harbor is not None:
                if self.board.corners[j][i].harbor == 'x':
                    for resource in resource_types:
                        self.players[player].trade_rates[resource] = min(self.players[player].trade_rates[resource], 3)
                else:
                    self.players[player].trade_rates[self.board.corners[j][i].harbor] = 2
            self.players[player].points += 1

        def city_place(player, coor):
            print(player, coor)
            i, j = coor
            assert self.board.corners[j][i].settlement == player
            self.board.corners[j][i].settlement = None
            self.board.corners[j][i].city = player
            self.players[player].tokens['settlement'] += 1
            self.players[player].tokens['city'] -= 1
            self.players[player].points += 1

        def development_place(player, coor):
            # dummy function
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
        # print('robber being moved')
        i, j = coor
        a, b = robber_coor
        self.board.hexes[j][i].robber=True
        self.board.hexes[b][a].robber=False
        self.board.robber = (i, j)
        self.instruction(f'{player} moved robber to {coor}')
        steal_choices = reduce(list.__add__, [[(p, resource)] * self.players[p].resource[resource] for p in filter(lambda x: x!= player, has_settlement(i, j)) for resource in resource_types], [])
        # print(steal_choices)
        to_steal = random.choice(steal_choices) if len(steal_choices) > 0 else None
        if to_steal is not None:
            self.players[to_steal[0]].resource[to_steal[1]] -= 1
            self.players[player].resource[to_steal[1]] += 1
            # self.instruction_label.config(text=f'{player} steals 1 {to_steal[1]} from {to_steal[0]}')
            self.instruction(f'{player} steals 1 {to_steal[1]} from {to_steal[0]}')
        return to_steal

    def player_pip(self, player):
        out = {resource: 0 for resource in ['ore', 'brick', 'wool', 'grain', 'lumber']}
        for corner in self.board.corner_list:
            if corner.settlement == player:
                for resource, pip in corner.resource_pips().items():
                    out[resource] += pip
            elif corner.city == player:
                for resource, pip in corner.resource_pips().items():
                    out[resource] += 2*pip
        return out

