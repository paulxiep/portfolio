from catan_session import *
from catan_units.catan_hex import *
from catan_units.catan_player import *
from catan_units.game_parameters import *
from catan_units.catan_ai import CatanAI
from catan_app import CatanApp
from itertools import groupby
from functools import reduce
import random
import logging
import os

def setup_logger(name, log_file, level=logging.INFO, format='state'):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(logging.Formatter(f'{format}: %(message)s'))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger
# action_logger = setup_logger('action_logger', 'log.txt')
# states_logger = setup_logger('states_logger', 'states.txt')
# logger.basicConfig()
class CatanSimulation(CatanSession):
    def __init__(self, players, board_type=4, actionlogfile='action_log.txt', statelogfile='states_log.txt', optionlogfile='option_log.txt'):
        CatanSession.__init__(self, players, board_type, actionlogfile=actionlogfile)
        self.state_logger = setup_logger(statelogfile, statelogfile)
        self.option_logger = setup_logger(optionlogfile, optionlogfile, format='option')
        # self.action_logger = setup_logger('action_logger', actionlogfile)
        # self.run_simulation()
        # CatanApp(players, board=self.board).mainloop()
        harbor_list = [x[0] for x in [((1, 2), (1, 1)), ((2, 1), (3, 2)), ((4, 3), (5, 4)),
                                      ((6, 6), (6, 7)), ((5, 9), (4, 10)), ((3, 11), (2, 12)),
                                      ((1, 12), (1, 11)), ((1, 9), (1, 8)), ((1, 5), (1, 4))]]
        # print(harbor_list)
        self.record_state(self.harbors)
        self.record_state(self.resources)
        self.record_state(self.numbers)
        self.actionlogfile = actionlogfile
        self.statelogfile = statelogfile
        self.optionlogfile = optionlogfile
        self.remove_log = False
    def record_state(self, log):
        self.state_logger.info(log)

    def record_option(self, options):
        self.option_logger.info(options)

    def run_simulation(self):
        while not self.game_over:
            self.game_state = next(self.game_state_generator)
            self.instruction(self.game_state)
            self.record_state(self.game_state)
            self.record_option(self.game_state)
            self.record_state([[corner.settlement, corner.city] for corner in self.board.corner_list])
            self.record_state([edge.road for edge in self.board.edge_list])
            self.record_state([hex.robber for hex in self.board.hex_list])
            for player in self.players.values():
                self.record_state(reduce(list.__add__, [list(player.resource.values()),
                                   list(player.development.values()),
                                   list(player.tokens.values()),
                                   list(player.trade_rates.values())]))
            current_player = self.players[self.game_state[1]]
            if self.game_state[2] == 'dice':
                while len(current_player.inactive_development) > 0:
                    development = current_player.inactive_development.pop()
                    current_player.development[development] += 1
                    if development == 'victory':
                        current_player.points += 1
                if current_player.development['knight'] > 0:
                    self.record_option(True)
                else:
                    self.record_option(False)
                if current_player.sub_ais['knight_ai'](self, current_player):
                    # self.record_option([hex.coor for hex in self.board.hex_list if hex.resource!='desert' and not hex.robber])
                    self.use_development(current_player.name, 'knight')
            elif self.game_state[2] == 'resolve_dice':
                self.dice = random.randrange(1, 7) + random.randrange(1, 7)
                self.instruction(f'{current_player.name} rolled {self.dice}')
                if self.dice != 7:
                    collect = {player: {resource: 0 for resource in resource_types} for player in self.players.keys()}
                    for corner in self.board.corner_list:
                        if corner.settlement is not None:
                            for hex in corner.hexes.values():
                                if hex is not None and hex.number == self.dice and not hex.robber:
                                    collect[corner.settlement][hex.resource] += 1
                                    self.players[corner.settlement].resource[hex.resource] += 1

                        if corner.city is not None:
                            for hex in corner.hexes.values():
                                if hex is not None and hex.number == self.dice and not hex.robber:
                                    collect[corner.city][hex.resource] += 2
                                    self.players[corner.city].resource[hex.resource] += 2

                    to_collect_text = ',\n'.join(list(filter(lambda x: not x.endswith('ects '), [
                        f'{player} collects ' + ', '.join([f'{v} {k}' for k, v in collect[player].items() if v > 0]) for
                        player in self.players.keys()])))
                    self.instruction(to_collect_text)
                else:
                    to_discard = {}
                    for player in self.players.keys():
                        total = reduce(list.__add__,
                                       [[resource] * self.players[player].resource[resource] for resource in
                                        resource_types])
                        if len(total) > 7:
                            to_discard[player] = sorted(random.sample(total, len(total) // 2))
                            for resource in to_discard[player]:
                                self.players[player].resource[resource] -= 1
                    to_discard_text = ',\n'.join(list(filter(lambda x: not x.endswith('ards '), [
                        f'{player} discards ' + ', '.join([f'{v} {k}' for k, v in {key: len(list(res)) for key, res in
                                                                                   groupby(
                                                                                       to_discard[player])}.items()])
                        for player in to_discard.keys()])))
                    self.instruction(to_discard_text)
                    self.moving_robber(current_player.name)
                self.record_option([current_player.development[dev]>0 for dev in ['road', 'plenty', 'monopoly']])
                for item in current_player.sub_ais['development_ai'](self, current_player):
                    self.use_development(current_player.name, item)
            elif self.game_state[2] == 'trade':
                self.record_option(reduce(list.__add__, map(lambda x: [(x, y) for y in current_player.trade_rates.keys() if y != x], [trade_away for trade_away, rate
                                    in current_player.trade_rates.items() if current_player.resource[trade_away]>=rate]), []))
                trades=current_player.sub_ais['trade_ai'](self, current_player)
                for trade in trades:
                    self.bank_trade(self.game_state[1], trade[0], trade[1])
            elif self.game_state[2] == 'build':
                self.record_option([(key, reduce(bool.__and__, [current_player.resource[resource]>=build_price[resource] for resource in build_price]), getattr(self, f'compile_eligible_{key}')(current_player)) for key, build_price in build_options.items()])
                builds = current_player.sub_ais['build_ai'](self, current_player)
                # print(builds)
                for build in builds:
                    self.instruction(f'{current_player.name} buys {build[0]}')
                    if self.buy(build[0], current_player.name):
                        self.instruction(f'{current_player.name} places {build[0]} at {build[1]}')
                        self.place(build[0])(current_player.name, build[1])
            elif self.game_state[2] == 'settlement':
                self.pending_process.append('place')
                self.record_option(self.compile_eligible_settlement_initial())
                if self.game_state[0] == 'setup_1':
                    coor = current_player.sub_ais['first_settlement_ai'](self, current_player)
                else:
                    coor = current_player.sub_ais['second_settlement_ai'](self, current_player)
                    to_collect = {resource: 0 for resource in resource_types}
                    for hex in self.board.corners[coor[1]][coor[0]].hexes.values():
                        if hex is not None and hex.resource != 'desert':
                            to_collect[hex.resource] += 1
                            current_player.resource[hex.resource] += 1
                self.instruction(f'{current_player.name} placed settlement at {coor}')
                self.place('settlement')(current_player.name, coor)
            elif self.game_state[2] == 'road':
                self.pending_process.append('place')
                self.record_option(self.compile_eligible_road_initial(current_player))
                coor = current_player.sub_ais['initial_road_ai'](self, current_player)
                self.instruction(f'{current_player.name} placed road at {coor}')
                self.place('road')(current_player.name, coor)
            elif self.game_state[2] == 'collect_resource':
                self.record_option([])

            if self.game_state[0] == 'turn_26':
                self.game_over = True
                self.record_state(f'time out - no winner')
                self.state_logger.removeHandler(self.state_logger.handlers[0])
                self.option_logger.removeHandler(self.option_logger.handlers[0])
                self.action_logger.removeHandler(self.action_logger.handlers[0])
                self.remove_log = True

            for player in self.players.keys():
                if self.players[player].points >= 10:
                    self.game_over = True
                    self.instruction(f'{player} wins')
                    self.record_state(f'winner is {player}')
                # for player in self.players.keys():
                #     self.record_state(self.players[player])
                #     for corner in self.board.corner_list:
                #         if corner.settlement == player:
                #             self.record_state(f'{player} has settlement at {corner.coor}')
                #         if corner.city == player:
                #             self.record_state(f'{player} has city at {corner.coor}')
                #     for edge in self.board.edge_list:
                #         if edge.road == player:
                #             self.record_state(f'{player} has road at {edge.coor}')
        if self.remove_log:
            os.remove(self.actionlogfile)
            os.remove(self.statelogfile)
            os.remove(self.optionlogfile)



