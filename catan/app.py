from catan_hex import *
from tkinter import *
from catan_canvas import CatanCanvas
import random
from functools import reduce, partial
from itertools import groupby
from gtts import gTTS
import playsound
# import os
# from time import sleep

player_order = ['white', 'orange', 'blue', 'red']
# ai_players = {'red': 'greedy_ai'}
ai_players = {player: 'greedy_ai' for player in player_order}
player_last = {player: None for player in player_order}
player_last_turn = {player: 0 for player in player_order}
first_actions = ['settlement', 'road']
second_actions = ['settlement', 'collect_resource', 'road']
turn_actions = ['dice', 'resolve_dice', 'trade', 'build']
game_state_dict = {'catan': {'initial_placement':
                   {'first_placement': {player: first_actions for player in player_order},
                    'second_placement': {player: second_actions for player in reversed(player_order)}},
               'main_session': {f'round_{i}': {player: turn_actions for player in player_order} for i in range(1, 1000)}
               }}
die = [1, 2, 3, 4, 5, 6]
resource_types = ['ore', 'brick', 'wool', 'grain', 'lumber']
auto_ai = True
build_options = {'road': {'brick': 1, 'lumber': 1},
                 'settlement': {'brick': 1, 'lumber': 1, 'wool': 1, 'grain': 1},
                 'city': {'ore': 3, 'grain': 2},
                 'development': {'ore': 1, 'grain': 1, 'wool': 1}}
# build_options = {'road': {}, 'settlement': {}, 'city': {}, 'development': {}}
development_cards = ['knight'] * 14 + ['road'] * 2 + ['plenty'] * 2 + ['monopoly'] * 2 + ['victory'] * 5

class CatanSession(Tk):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        self.title('Catan')
        self.geometry('1200x600')
        self.canvas_frame = Frame(self, width=500, height=600)
        self.canvas = CatanCanvas(self.canvas_frame, self, width=500, height=600)
        self.canvas.place(x=0, y=0)
        self.board = CatanHexBoard(4)
        self.draw_board()
        self.canvas_frame.grid(row=0, column=0)
        self.ui_frame = Frame(self, width=700, height=600)
        self.ui_frame.grid(row=0, column=1)
        self.state_label = Label(self, text='Catan', font='Helvetica 18 bold')
        self.state_label.place(x=10, y=10)
        self.instruction_label = Label(self, text='Click Proceed', font='Helvetica 16')
        self.instruction_label.place(x=610, y=50)
        self.generic_ui_frame = Frame(self.ui_frame, width=700, height=100)
        self.turn_ui_frame = Frame(self.ui_frame, width=700, height=500)
        self.generic_ui_frame.grid(row=1, column=0)
        self.turn_ui_frame.grid(row=0, column=0)
        self.continue_button = Button(self.generic_ui_frame, text='Proceed', command=self.setup)
        self.continue_button.place(x=300, y=20)
        self.game_state = []
        self.players = {player: {
            'resource': 0,
            'ore': 0,
            'brick': 0,
            'lumber': 0,
            'grain': 0,
            'wool': 0,
            'army': 0,
            'development': {
                'knight': 0,
                'plenty': 0,
                'road': 0,
                'monopoly': 0,
                'victory': 0,
                'development': 0
            }
        } for player in player_order}
        self.tokens = {player: {'settlement': 5, 'city': 4, 'road': 5} for player in player_order}
        self.inactive_development = {player:[] for player in player_order}
        self.development_pool = development_cards.copy()
        self.resource_data = {player: {} for player in player_order}
        self.resource_spin = {player: {} for player in player_order}
        self.player_columns = {}
        self.resource_rows = {}
        self.trade_rates = {player: {resource: 4 for resource in resource_types} for player in player_order}
        self.wait_move_robber = False
        self.wait_build = False
        self.wait_process = False
        self.process_list = []
        self.discarding = False
        self.plenty = False
        self.monopoly = False
        self.wait_ai = False

    def instruction(self, instruction):
        print(instruction)
        self.instruction_label.config(text=instruction)
        try:
            gTTS(text=instruction, lang='en').save('temp.mp3')
            playsound.playsound('.\\temp.mp3')
        except:
            return None

    def clear_ui_frame(self):
        for widget in self.generic_ui_frame.winfo_children():
            widget.destroy()
        for widget in self.turn_ui_frame.winfo_children():
            widget.destroy()
        print('ui cleared')
    def dummy(self):
        print('dummy')
    def exit(self):
        self.continue_var.set(True)
        self.destroy()
    def setup(self):
        print('setup board')
        self.board.assign_resources()
        self.board.assign_numbers()
        self.canvas.delete('all')
        self.draw_board()
        self.clear_ui_frame()
        self.continue_var = BooleanVar(False)
        self.continue_button = Button(self.generic_ui_frame, text='Proceed', command=self.start_session)
        self.continue_button.place(x=300, y=20)
        self.dummy_button = Button(self.generic_ui_frame, text='Dummy', command=self.dummy)
        self.dummy_button.place(x=200, y=20)
        self.quit_button = Button(self.generic_ui_frame, text='Exit', command=self.exit)
        self.quit_button.place(x=400, y=20)
        self.update()
    def start_session(self):
        print('starting_session')
        self.clear_ui_frame()
        self.continue_var = BooleanVar(False)
        self.continue_button = Button(self.generic_ui_frame, text='Proceed', command=self.advance_state)
        self.continue_button.place(x=300, y=20)
        self.dummy_button = Button(self.generic_ui_frame, text='Dummy', command=self.dummy)
        self.dummy_button.place(x=200, y=20)
        self.quit_button = Button(self.generic_ui_frame, text='Exit', command=self.exit)
        self.quit_button.place(x=400, y=20)
        self.table = Frame(self.turn_ui_frame, width=300)
        self.table.place(x=20, y=150)
        self.build_buttons = {build: Button(self.turn_ui_frame, text=f'buy {build}', command=getattr(self, f'buy_{build}'), state='disabled') for build in build_options.keys()}
        [build_button.place(x=20+100*i, y=450) for i, build_button in enumerate(self.build_buttons.values())]
        Label(self.turn_ui_frame, text='trade away').place(x=260, y=170)
        Label(self.turn_ui_frame, text='trade for').place(x=400, y=170)
        self.trade_button = Button(self.turn_ui_frame, text='Trade', command=self.bank_trade, state='disabled')
        self.trade_button.place(x=345, y=200)
        self.trade_away_var = StringVar(self)
        self.trade_away_option = OptionMenu(self.turn_ui_frame, self.trade_away_var, *resource_types)
        self.trade_away_option.place(x=260, y=200)
        self.trade_for_var = StringVar(self)
        self.trade_for_option = OptionMenu(self.turn_ui_frame, self.trade_for_var, *resource_types)
        self.trade_for_option.place(x=400, y=200)
        self.development_var = StringVar(self)
        Label(self.turn_ui_frame, text='development').place(x=260, y=380)
        self.development_option = OptionMenu(self.turn_ui_frame, self.development_var, *[card for card in set(development_cards) if card != 'victory'])
        self.development_option.place(x=260, y=400)
        self.use_development_button = Button(self.turn_ui_frame, text='Use', command=self.use_development)
        self.use_development_button.place(x=360, y=400)

        for j, resource in enumerate(self.players[player_order[0]].keys()):
            if not isinstance(self.players[player_order[0]][resource], dict):
                self.resource_rows[resource] = Label(self.table, text=resource)
                self.resource_rows[resource].grid(row=j+1, column=0)
            else:
                for k, development in enumerate(self.players[player_order[0]][resource].keys()):
                    self.resource_rows[resource] = Label(self.table, text=development)
                    self.resource_rows[resource].grid(row=j+1+k, column=0)
        for i, player in enumerate(self.players.keys()):
            self.player_columns[player] = Label(self.table, text=player)
            self.player_columns[player].grid(row=0, column=i+1)
            for j, resource in enumerate(self.players[player].keys()):
                if not isinstance(self.players[player][resource], dict):
                    self.resource_data[player][resource] = IntVar(self, value=0)
                    self.resource_spin[player][resource] = Spinbox(self.table, values=tuple(range(20)), width=4, textvariable=self.resource_data[player][resource], state='disabled')
                    self.resource_spin[player][resource].grid(row=j+1, column=i+1)
                else:
                    for k, development in enumerate(self.players[player][resource].keys()):
                        self.resource_data[player][development] = IntVar(self, value=0)
                        self.resource_spin[player][development] = Spinbox(self.table, values=tuple(range(20)), width=4, textvariable=self.resource_data[player][development], state='disabled')
                        self.resource_spin[player][development].grid(row=j+1+k, column=i+1)
        self.update()
        self.dfs(game_state_dict, 'catan')

    def compile_ineligible_settlement(self):
        ineligible = []
        for j, row in self.board.corners.items():
            for i, corner in row.items():
                if self.board.corners[j][i].settlement is not None:
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

    def compile_eligible_road(self):
        for player in player_order:
            if player in self.game_state:
                break
        eligible = []
        for corner in self.board.corner_list:
            if corner.settlement == player or corner.city == player:
                eligible += [edge for edge in corner.edges.values()]
        for edge in self.board.edge_list:
            if edge.road==player:
                eligible += reduce(list.__add__, [[edge2 for edge2 in corner.edges.values() if edge2 is not None] for corner in edge.corners.values() if corner is not None and ((corner.settlement is None and corner.city is None) or corner.settlement == player or corner.city == player)])
        #     eligible += [edge2 for edge2 in edge.edges.values() if edge.road == player if edge2 is not None]
        eligible = [edge.coor for edge in eligible if edge is not None]
        return list(filter(lambda x: self.board.edges[x[1]][x[0]].road is None, eligible))

    def compile_eligible_settlement(self):
        for player in player_order:
            if player in self.game_state:
                break
        eligible = []
        for edge in self.board.edge_list:
            # [print(edge.coor, edge.type, corner.coor) for corner in edge.corners.values() if edge.road == player]
            eligible += [corner.coor for corner in edge.corners.values() if edge.road == player]
        return [corner for corner in eligible if corner not in self.compile_ineligible_settlement()]

    def compile_eligible_city(self):
        for player in player_order:
            if player in self.game_state:
                break
        eligible = []
        for corner in self.board.corner_list:
            if corner.settlement == player:
                eligible += [corner.coor]
        return eligible

    def create_turn_buttons(self):
        def compile_eligible_road_initial():
            for player in player_order:
                if player in self.game_state:
                    break
            last_corner = player_last[player]
            eligible = [edge.coor for edge in last_corner.edges.values() if edge is not None]
            return eligible
        for current_player in player_order:
            if current_player in self.game_state:
                break
        if current_player in ai_players.keys():
            self.wait_ai = True
        else:
            self.wait_ai = False
        if self.game_state[-1] == current_player:
            while len(self.inactive_development[current_player]) > 0:
                development = self.inactive_development[current_player].pop()
                self.players[current_player]['development'][development] += 1
                self.resource_data[current_player][development].set(self.players[current_player]['development'][development])
        if self.game_state[-1]=='road':
            # self.instruction_label.config(text=f'{current_player} place road')
            self.instruction(f'{current_player} place road')
            if 'initial_placement' in self.game_state:
                eligible = compile_eligible_road_initial()
            self.canvas.draw_road_placer(self.board, self, eligible=eligible)
            self.update()
        elif self.game_state[-1]=='settlement':
            # self.instruction_label.config(text=f'{current_player} place settlement')
            self.instruction(f'{current_player} place settlement')
            if 'initial_placement' in self.game_state:
                ineligible = self.compile_ineligible_settlement()
            self.canvas.draw_buttons(self.board, self, ineligible=ineligible)
            self.update()
        elif self.game_state[-1] == 'collect_resource' and 'initial_placement' in self.game_state:
            to_collect = {resource: 0 for resource in resource_types}
            for hex in player_last[current_player].hexes.values():
                if hex is not None and hex.resource != 'desert':
                    to_collect[hex.resource] += 1
                    self.players[current_player][hex.resource] += 1
                    self.resource_data[current_player][hex.resource].set(self.players[current_player][hex.resource])
            to_collect_text = ' and '.join([f'{v} {k}' for k, v in to_collect.items() if v>0])
            # self.instruction_label.config(text=f'{current_player} collects {to_collect_text}')
            self.instruction(f'{current_player} collects {to_collect_text}')
        elif self.game_state[-1] == 'dice':
            # self.instruction_label.config(text=f'{current_player} throws dice')
            self.instruction(f'{current_player} throws dice')
        elif self.game_state[-1] == 'resolve_dice' and self.dice!=7:
            collect = {player: {resource: 0 for resource in resource_types} for player in player_order}
            for corner in self.board.corner_list:
                if corner.settlement is not None:
                    for hex in corner.hexes.values():
                        if hex is not None and hex.number == self.dice and not hex.robber:
                            collect[corner.settlement][hex.resource] += 1
                            self.players[corner.settlement][hex.resource] += 1
                            self.resource_data[corner.settlement][hex.resource].set(self.players[corner.settlement][hex.resource])

                if corner.city is not None:
                    for hex in corner.hexes.values():
                        if hex is not None and hex.number == self.dice and not hex.robber:
                            collect[corner.city][hex.resource] += 2
                            self.players[corner.city][hex.resource] += 2
                            self.resource_data[corner.city][hex.resource].set(self.players[corner.city][hex.resource])

            to_collect_text = ',\n'.join(list(filter(lambda x: not x.endswith('ects '), [f'{player} collects ' + ', '.join([f'{v} {k}' for k, v in collect[player].items() if v > 0]) for player in player_order])))
            if len(to_collect_text)==0:
                to_collect_text = 'Click Proceed'
            # self.instruction_label.config(text=to_collect_text)
            self.instruction(to_collect_text)
        elif self.game_state[-1] == 'resolve_dice' and self.dice==7:
            to_discard = {}
            for player in player_order:
                total = reduce(list.__add__, [[resource] * self.players[player][resource] for resource in resource_types])
                if len(total) > 7:
                    to_discard[player] = random.sample(total, len(total)//2)
                    for resource in to_discard[player]:
                        self.players[player][resource] -= 1
                        self.resource_data[player][resource].set(self.players[player][resource])
            to_discard_text = ',\n'.join(list(filter(lambda x: not x.endswith('ards '), [f'{player} discards ' + ', '.join([f'{v} {k}' for k, v in {key: len(list(res)) for key, res in groupby(to_discard[player])}.items()]) for player in to_discard.keys()])))
            if len(to_discard_text)==0:
                to_discard_text = 'Click proceed after moving robber'
            # self.instruction_label.config(text=to_discard_text)
            self.instruction(to_discard_text)
            # print(to_discard_text)
            self.moving_robber()
        else:
            self.instruction_label.config(text='Click Proceed')
        self.trade_button.config(state='disabled')
        [build_button.config(state='disabled') for build_button in self.build_buttons.values()]
        if self.game_state[-1] == 'build':
            [build_button.config(state='normal') for build_button in self.build_buttons.values()]
        elif self.game_state[-1] == 'trade':
            self.trade_button.config(state='normal')
        if self.game_state[-1] == 'trade' or self.game_state[-1] == 'resolve_dice':
            [self.resource_spin[p][key].config(state='normal') for p in player_order for key in resource_types]
        else:
            [self.resource_spin[p][key].config(state='disabled') for p in player_order for key in resource_types]
            for p in player_order:
                for key in resource_types:
                    self.players[p][key] = self.resource_data[p][key].get()

    def bank_trade(self, trade_away=None, trade_for=None):
        for player in player_order:
            if player in self.game_state:
                break
        if trade_away is None and trade_for is None:
            if self.trade_away_var.get() == '' or self.trade_for_var.get() == '':
                print('trade option empty')
                self.instruction_label.config(text=f'trade option empty')
                return None
            trade_away = self.trade_away_var.get()
            trade_for = self.trade_for_var.get()
        if self.players[player][trade_away] < self.trade_rates[player][trade_away]:
            print(f'not enough {trade_away} for trade')
            self.instruction_label.config(text=f'not enough {trade_away} for trade')
            return None
        else:
            self.players[player][trade_away] -= self.trade_rates[player][trade_away]
            self.players[player][trade_for] += 1
            # self.instruction_label.config(text=f'exchanged {trade_away} for {trade_for}')
            self.instruction(f'exchanged {trade_away} for {trade_for}')
            self.resource_data[player][trade_away].set(self.players[player][trade_away])
            self.resource_data[player][trade_for].set(self.players[player][trade_for])

    def buy_road(self, free=False):
        eligible = self.compile_eligible_road()
        if len(eligible) == 0:
            print(f'no eligible road placement')
            self.instruction_label.config(text=f'no eligible road placement')
            return None
        for player in player_order:
            if player in self.game_state:
                break
        if not free:
            for k, v in build_options['road'].items():
                if self.players[player][k] < v:
                    print(f'not enough resource to purchase road')
                    self.instruction_label.config(text=f'not enough resource to purchase road')
                    return None
            for k, v in build_options['road'].items():
                self.players[player][k] -= v
                self.resource_data[player][k].set(self.players[player][k])
        # self.instruction_label.config(text=f'{player} place road')
        self.instruction(f'{player} place road')
        self.canvas.draw_road_placer(self.board, self, eligible=eligible)
        self.update()
        self.wait_build = True
    def buy_settlement(self):
        eligible = self.compile_eligible_settlement()
        if len(eligible) == 0:
            print(f'no eligible settlement placement')
            self.instruction_label.config(text=f'no eligible settlement placement')
            return None
        for player in player_order:
            if player in self.game_state:
                break
        for k, v in build_options['settlement'].items():
            if self.players[player][k] < v:
                print(f'not enough resource to purchase settlement')
                self.instruction_label.config(text=f'not enough resource to purchase settlement')
                return None
        for k, v in build_options['settlement'].items():
            self.players[player][k] -= v
            self.resource_data[player][k].set(self.players[player][k])
        # self.instruction_label.config(text=f'{player} place settlement')
        self.instruction(f'{player} place settlement')
        self.canvas.draw_buttons(self.board, self, eligible=eligible)
        self.update()
        self.wait_build = True
    def buy_city(self):
        eligible = self.compile_eligible_city()
        if len(eligible) == 0:
            print(f'no eligible city placement')
            self.instruction_label.config(text=f'no eligible city placement')
            return None
        for player in player_order:
            if player in self.game_state:
                break
        for k, v in build_options['city'].items():
            if self.players[player][k] < v:
                print(f'not enough resource to purchase city')
                self.instruction_label.config(text=f'not enough resource to purchase city')
                return None
        for k, v in build_options['city'].items():
            self.players[player][k] -= v
            self.resource_data[player][k].set(self.players[player][k])
        # self.instruction_label.config(text=f'{player} place city')
        self.instruction(f'{player} place city')
        self.canvas.draw_buttons(self.board, self, eligible=eligible)
        self.update()
        self.wait_build=True
    def buy_development(self):
        if len(self.development_pool)==0:
            print(f'no more development card to purchase')
            self.instruction_label.config(text=f'no more development card to purchase')
            return None
        for player in player_order:
            if player in self.game_state:
                break
        for k, v in build_options['development'].items():
            if self.players[player][k] < v:
                print(f'not enough resource to purchase development')
                self.instruction_label.config(text=f'not enough resource to purchase development')
                return None
        for k, v in build_options['development'].items():
            self.players[player][k] -= v
            self.resource_data[player][k].set(self.players[player][k])
        self.inactive_development[player].append(self.development_pool.pop(random.randrange(len(self.development_pool))))
        # self.instruction_label.config(text=f'{player} purchased development')
        self.instruction(f'{player} purchased development')
        self.update()
    def use_plenty(self, player):
        # self.instruction_label.config(text=f'{player} take free resource')
        self.instruction(f'{player} take free resource')
        self.canvas.draw_resource_buttons(self.board)
        self.update()
    def use_monopoly(self, player):
        # self.instruction_label.config(text=f'{player} choose resource to monopolize')
        self.instruction(f'{player} choose resource to monopolize')
        self.canvas.draw_resource_buttons(self.board)
        self.update()
    def use_development(self):
        def use_knight():
            self.players[player]['army'] += 1
            self.resource_data[player]['army'].set(self.players[player]['army'])
            self.moving_robber()
        def use_road():
            self.wait_process=True
            self.process_list.append('road')
            self.buy_road(free=True)
        def use_plenty():
            self.plenty=True
            self.wait_process=True
            self.process_list.append('plenty')
            self.use_plenty(player)
        def use_monopoly():
            self.monopoly=True
            self.wait_process=True
            self.use_monopoly(player)
        for player in player_order:
            if player in self.game_state:
                break
        development = self.development_var.get()
        if development == '':
            print(f"choose development card to use")
            self.instruction_label.config(text=f"choose development card to use")
            return None
        if development == 'victory':
            print(f"victory cards can't be played")
            self.instruction_label.config(text=f"victory cards can't be played")
            return None
        if self.players[player]['development'][development] < 1:
            print(f'no {development} card to use')
            self.instruction_label.config(text=f'no {development} card to use')
            return None
        self.players[player]['development'][development] -= 1
        self.resource_data[player][development].set(self.players[player]['development'][development])

        locals()[f'use_{development}']()

        self.update()
    def moving_robber(self):
        for player in player_order:
            if player in self.game_state:
                break
        # self.instruction_label.config(text=f'{player} move robber')
        self.instruction(f'{player} move robber')
        self.canvas.draw_robber_buttons(self.board, self)
        self.update()
        self.wait_move_robber = True
        if player in ai_players.keys():
            print(player, 'is ai, calling robber ai')
            self.robber_ai()
    def choose_resource(self, resource):
        for current_player in player_order:
            if current_player in self.game_state:
                break
        if self.plenty:
            self.players[current_player][resource] += 1
            self.resource_data[current_player][resource].set(self.players[current_player][resource])
            if len(self.process_list) > 0 and self.process_list.pop(0)=='plenty':
                self.use_plenty(current_player)
            else:
                self.plenty=False
                self.wait_process=False
                self.canvas.delete('all')
                self.draw_board()
        elif self.monopoly:
            take_text = []
            for player in player_order:
                if player != current_player:
                    resource_count = self.players[player][resource]
                    if resource_count > 0:
                        self.players[player][resource] = 0
                        self.resource_data[player][resource].set(self.players[player][resource])
                        self.players[current_player][resource] += resource_count
                        self.resource_data[current_player][resource].set(self.players[current_player][resource])
                        take_text += [f'{current_player} takes {resource_count} {resource} from {player}']
            self.monopoly = False
            self.wait_process = False
            # self.instruction_label.config(text='\n'.join(take_text))
            if len(take_text)>0:
                self.instruction(',\n'.join(take_text))
            self.canvas.delete('all')
            self.draw_board()

    def road_place(self, i, j):
        print('road being placed')
        for player in player_order:
            if player in self.game_state:
                break
        self.board.edges[j][i].road=player
        self.tokens[player]['road'] -= 1
        player_last[player] = self.board.edges[j][i]
        self.canvas.delete('all')
        self.draw_board()
        self.wait_build=False
        if self.wait_process and self.process_list[0]=='road':
            self.process_list.pop(0)
            self.wait_process=False
            self.buy_road(free=True)
    def corner_place(self, i, j):
        for player in player_order:
            if player in self.game_state:
                break
        if self.game_state[-1] == 'settlement':
            self.board.corners[j][i].settlement=player
            player_last[player] = self.board.corners[j][i]
            self.tokens[player]['settlement'] -= 1
        elif self.board.corners[j][i].settlement is None:
            self.board.corners[j][i].settlement=player
            self.tokens[player]['settlement'] -= 1
        else:
            self.board.corners[j][i].settlement=None
            self.board.corners[j][i].city=player
            self.tokens[player]['settlement'] += 1
            self.tokens[player]['city'] -= 1
        if self.board.corners[j][i].harbor is not None:
            if self.board.corners[j][i].harbor == 'x':
                for resource in resource_types:
                    self.trade_rates[player][resource] = min(self.trade_rates[player][resource], 3)
            else:
                self.trade_rates[player][self.board.corners[j][i].harbor] = 2
        self.canvas.delete('all')
        self.draw_board()
        self.wait_build = False
    def resolve_state(self):
        for player in player_order:
            if player in self.game_state:
                if self.game_state[-1]==player:
                    return True
                else:
                    print('resolving actions')
                    if self.game_state[-1]=='settlement' and not isinstance(player_last[player], CatanCorner):
                        if auto_ai or player in ai_players.key():
                            self.settlement_ai()
                            return True
                        print('place settlement first')
                        return False
                    elif self.game_state[-1]=='road' and not isinstance(player_last[player], CatanEdge):
                        if auto_ai or player in ai_players.keys():
                            self.road_ai()
                            return True
                        print('place road first')
                        return False
                    elif self.wait_move_robber==True:
                        print('move robber first')
                        return False
                    elif self.wait_build==True:
                        print('finish building first')
                        return False
                    elif self.wait_process==True:
                        print('resolve current process')
                        return False
                    elif self.game_state[-1]=='build':
                        if self.wait_ai:
                            self.build_ai()
                            self.wait_ai=False
                            return False
                        return True
                    elif self.game_state[-1]=='trade':
                        if self.wait_ai:
                            self.trade_ai()
                            self.wait_ai=False
                            return False
                        # self.build_ai()
                        return True
                    elif self.game_state[-1]=='dice':
                        if self.wait_ai:
                            self.knight_ai()
                            self.wait_ai=False
                            return False
                        self.dice = random.choice(die) + random.choice(die)
                        print(f'{player} rolled {self.dice}')
                        return True
                    elif self.game_state[-1]=='resolve_dice':
                        if self.wait_ai:
                            self.development_ai()
                            self.wait_ai=False
                            return False
                        return True
                    else:
                        return True
        return True
    def move_robber(self, i, j, a, b):
        def has_settlement(i, j):
            out = []
            for corner in self.board.hexes[j][i].corners.values():
                if corner is not None:
                    if corner.settlement is not None:
                        out.append(corner.settlement)
                    elif corner.city is not None:
                        out.append(corner.city)
            return list(set(out))
        print('robber being moved')
        for player in player_order:
            if player in self.game_state:
                break
        self.board.hexes[j][i].robber=True
        self.board.hexes[b][a].robber=False
        steal_choices = reduce(list.__add__, [[(p, resource)] * self.players[p][resource] for p in filter(lambda x: x!= player, has_settlement(i, j)) for resource in resource_types], [])
        # print(steal_choices)
        to_steal = random.choice(steal_choices) if len(steal_choices) > 0 else None
        if to_steal is not None:
            self.players[to_steal[0]][to_steal[1]] -= 1
            self.players[player][to_steal[1]] += 1
            self.resource_data[to_steal[0]][to_steal[1]].set(self.players[to_steal[0]][to_steal[1]])
            self.resource_data[player][to_steal[1]].set(self.players[player][to_steal[1]])
            # self.instruction_label.config(text=f'{player} steals 1 {to_steal[1]} from {to_steal[0]}')
            self.instruction(f'{player} steals 1 {to_steal[1]} from {to_steal[0]}')
        # player_last[player] = self.board.edges[j][i]
        self.canvas.delete('all')
        self.draw_board()
        self.wait_move_robber = False
    def advance_state(self):
        if self.resolve_state():
            self.continue_var.set(True)
        else:
            print('resolve all actions before proceeding')
    def draw_board(self):
        self.canvas.draw_board(self.board)
    def dfs(self, graph, node):
        def continue_from_list(node):
            self.game_state.append(node)
            continue_event(node)
            self.game_state.pop(-1)
        # def continue_helper(node):
        def continue_from_dict(node):
            [self.dfs(graph[node], key) for key in graph[node].keys()]
        def continue_event(node):
            print(node)
            print(self.game_state)
            # self.title('-'.join(self.game_state))
            self.state_label.config(text='-'.join(self.game_state))
            self.create_turn_buttons()
            self.continue_button.wait_variable(self.continue_var)
            self.continue_var.set(False)
        self.game_state.append(node)
        continue_event(node)
        if isinstance(graph[node], dict):
            continue_from_dict(node)
        elif isinstance(graph[node], list):
            [continue_from_list(elem) for elem in graph[node]]
        else:
            print('error')
        self.game_state.pop(-1)
    def player_pip(self, player=None):
        if player is None:
            for player in player_order:
                if player in self.game_state:
                    break
        out = {resource: 0 for resource in ['ore', 'brick', 'wool', 'grain', 'lumber']}
        for corner in self.board.corner_list:
            if corner.settlement == player:
                for resource, pip in corner.resource_pips().items():
                    out[resource] += pip
            elif corner.city == player:
                for resource, pip in corner.resource_pips().items():
                    out[resource] += 2*pip
        return out
    def development_ai(self):
        for player in player_order:
            if player in self.game_state:
                break
        if self.players[player]['development']['road'] > 0:
            self.players[player]['development']['road'] -= 1
            self.resource_data[player]['road'].set(self.players[player]['development']['road'])
            self.road_ai(self.compile_eligible_road())
            self.road_ai(self.compile_eligible_road())
        if self.players[player]['development']['monopoly'] > 0:
            max_gain = max([(resource, self.players[p][resource]) for resource in resource_types for p in player_order if p!=player], key=lambda x: x[1])
            if max_gain[1] > len(player_order):
                self.players[player]['development']['monopoly'] -= 1
                self.resource_data[player]['monopoly'].set(self.players[player]['development']['monopoly'])
                self.monopoly = True
                self.choose_resource(max_gain[0])
        for build_option in ['city', 'settlement', 'road']:
            if self.players[player]['development']['plenty'] > 0:
                if len(getattr(self, f'compile_eligible_{build_option}')()) > 0:
                    budget = {k: min(self.players[player][k] - v, 0) for k, v in build_options[build_option].items()}
                    if -2 <= reduce(int.__add__, list(budget.values())) < 0:
                        self.players[player]['development']['plenty'] -= 1
                        self.resource_data[player]['plenty'].set(self.players[player]['development']['plenty'])

                        self.plenty = True
                        self.wait_process = True
                        self.process_list.append('plenty')
                        for k, v in budget.items():
                            if v == -2:
                                self.choose_resource(k)
                                self.choose_resource(k)
                            elif v == -1:
                                self.choose_resource(k)
                                self.choose_resource(min([(r, p) for r, p in self.player_pip(player).items()], key=lambda x: x[1] + random.random())[0])
                        return None
    def knight_ai(self):
        for player in player_order:
            if player in self.game_state:
                break
        if self.players[player]['development']['knight']==0:
            return None
        for hex in self.board.hex_list:
            if hex.robber and hex.resource!='desert':
                for corner in hex.corners.values():
                    if corner is not None and (corner.settlement == player or corner.city == player):
                        self.players[player]['development']['knight'] -= 1
                        self.resource_data[player]['knight'].set(self.players[player]['development']['knight'])
                        self.players[player]['army'] += 1
                        self.resource_data[player]['army'].set(self.players[player]['army'])
                        self.moving_robber()
                return None
    def robber_ai(self):
        def hex_score(hex):
            if hex.robber or hex.resource=='desert':
                return -1000
            pips = hex.pips()
            return reduce(int.__add__, [pips*(corner.settlement is not None and corner.settlement!=player) +\
             2*pips*(corner.city is not None and corner.city!=player) - \
            (100 * pips * (corner.settlement is not None and corner.settlement == player) + \
             200 * pips * (corner.city is not None and corner.city == player))
             for corner in hex.corners.values() if corner is not None])
        for player in player_order:
            if player in self.game_state:
                break
        for hex in self.board.hex_list:
            if hex.robber:
                a, b = hex.coor
        i, j = max(self.board.hex_list, key=hex_score).coor
        self.move_robber(i, j, a, b)
    def city_ai(self, eligible):
        coor_choice = max(eligible, key=lambda coor: self.board.corners[coor[1]][coor[0]].pips()+random.random())
        self.corner_place(coor_choice[0], coor_choice[1])
    def settlement_ai(self, eligible=None):
        # for current_player in player_order:
        #     if current_player in self.game_state:
        #         break
        if eligible is None:
            settlement_choices = [(x, y) for y, row in self.canvas.buttons.items() for x, b in row.items() if b is not None]
        else:
            settlement_choices = eligible
        coor_choice = max(settlement_choices, key=lambda coor: self.board.corners[coor[1]][coor[0]].pips()+random.random())
        self.corner_place(coor_choice[0], coor_choice[1])
    def road_ai(self, eligible=None):
        for current_player in player_order:
            if current_player in self.game_state:
                break
        eligible_settlements = self.compile_eligible_settlement()
        if eligible is None:
            road_choices = [(x, y) for y, row in self.canvas.edges.items() for x, b in row.items() if b is not None]
        else:
            road_choices = eligible
        gainful_choices = []
        for choice in road_choices:
            self.board.edges[choice[1]][choice[0]].road = current_player
            new = list(set(self.compile_eligible_settlement())-set(eligible_settlements))
            if len(new) > 0:
                gainful_choices.append((choice, self.board.corners[new[0][1]][new[0][0]].pips()))
            self.board.edges[choice[1]][choice[0]].road = None
        if len(gainful_choices) > 0:
            coor_choice = max(gainful_choices, key=lambda x: x[1]+random.random())[0]
        else:
            coor_choice = random.choice(road_choices)
        self.road_place(coor_choice[0], coor_choice[1])
    def trade_ai(self):
        def trade_weight(resource, required, trade_rates):
            amount = self.players[current_player][resource]
            if resource in required.keys():
                amount -= required[resource]
            amount -= trade_rates[resource]
            if amount < 0:
                return -100
            else:
                return amount + random.random()/2 + (4-trade_rates[resource])/4
        def try_trade(build_option):
            resource = max([resource for resource in resource_types], key=lambda x: trade_weight(x, build_options['city'], trade_rates))
            if trade_weight(resource, build_options[build_option], trade_rates) > 0:
                self.bank_trade(resource, [k for k, v in budget.items() if v<0][0])
                return True
            return False
        self.wait_ai = False
        for current_player in player_order:
            if current_player in self.game_state:
                break
        trade_rates = self.trade_rates[current_player]
        eligible_cities = self.compile_eligible_city()
        budget = {k: self.players[current_player][k] - v for k, v in build_options['city'].items()}
        if len(eligible_cities)>0:
            lacking = reduce(int.__add__, [min(v, 0) for v in budget.values()])
            if lacking == 0 or (lacking == -1 and try_trade('city')):
                return None
        eligible_settlements = self.compile_eligible_settlement()
        budget = {k: self.players[current_player][k] - v for k, v in build_options['settlement'].items()}
        if len(eligible_settlements)>0 and reduce(int.__add__, [min(v, 0) for v in budget.values()]) == -1:
            lacking = reduce(int.__add__, [min(v, 0) for v in budget.values()])
            if lacking == 0 or (lacking == -1 and try_trade('settlement')):
                return None
        eligible_roads = self.compile_eligible_road()
        budget = {k: self.players[current_player][k] - v for k, v in build_options['road'].items()}
        if len(eligible_roads)>0 and reduce(int.__add__, [min(v, 0) for v in budget.values()]) == -1:
            lacking = reduce(int.__add__, [min(v, 0) for v in budget.values()])
            if lacking == 0 or (lacking == -1 and try_trade('road')):
                return None
        budget = {k: self.players[current_player][k] - v for k, v in build_options['development'].items()}
        if len(eligible_roads)>0 and reduce(int.__add__, [min(v, 0) for v in budget.values()]) == -1:
            lacking = reduce(int.__add__, [min(v, 0) for v in budget.values()])
            if lacking == 0 or (lacking == -1 and try_trade('development')):
                return None

    def build_ai(self):
        self.wait_ai = False
        for current_player in player_order:
            if current_player in self.game_state:
                break
        eligible_cities = self.compile_eligible_city()
        save_for_city = False
        if self.tokens[current_player]['city'] > 0 and len(eligible_cities) > 0:
            required = [min(self.players[current_player][k] - v, 0) for k, v in build_options['city'].items()]
            if reduce(int.__add__, required) == 0:
                for k, v in build_options['city'].items():
                    self.players[current_player][k] -= v
                    self.resource_data[current_player][k].set(self.players[current_player][k])
                self.city_ai(eligible=eligible_cities)
                print(f'{current_player} built city')
            elif reduce(int.__add__, required) >= -1:
                save_for_city= True
                print(f'{current_player} saving up for city')
        eligible_settlements = self.compile_eligible_settlement()
        if self.tokens[current_player]['settlement'] > 0 and len(eligible_settlements) > 0:
            required = [self.players[current_player][k] >= v for k, v in build_options['settlement'].items()]
            if all(required):
                for k, v in build_options['settlement'].items():
                    self.players[current_player][k] -= v
                    self.resource_data[current_player][k].set(self.players[current_player][k])
                self.settlement_ai(eligible=eligible_settlements)
                print(f'{current_player} built settlement')
            if reduce(lambda x, y: x + int(y), required, 0) == 3:
                print(f'{current_player} saving up for settlement')
                return None
        # else:
        eligible_roads = self.compile_eligible_road()
        if self.tokens[current_player]['road'] > 0 and len(eligible_roads) > 0:
            required = [self.players[current_player][k] >= v for k, v in build_options['road'].items()]
            if all(required):
                for k, v in build_options['road'].items():
                    self.players[current_player][k] -= v
                    self.resource_data[current_player][k].set(self.players[current_player][k])
                self.road_ai(eligible=eligible_roads)
                print(f'{current_player} built road')
        if not save_for_city and len(self.development_pool)>0:
            required = [self.players[current_player][k] >= v for k, v in build_options['development'].items()]
            if all(required):
            #     for k, v in build_options['development'].items():
            #         self.players[current_player][k] -= v
            #         self.resource_data[current_player][k].set(self.players[current_player][k])
                self.buy_development()
                # print(f'{current_player} purchased development')



# def button_function(y, x):
#     def print_button(event):
#         print(f'button clicked {y} {x}')
#     return lambda event: print_button(event)




if __name__ == '__main__':
    root = CatanSession()
    # root.title('Catan')
    # root.geometry('800x600')
    # catan_board = CatanHexBoard(4)
    # for corner in catan_board.corner_list:
    #     corner.city = random.choice(['orange red', 'blue', 'floral white', None])
    # catan_canvas = CatanCanvas(root, width=600, height=600)
    # catan_canvas.draw_board(catan_board)
    # catan_canvas.place(x=0, y=0)
    # place_corner_button(root, catan_board)
    # main_thread = Thread(target=root.mainloop)
    root.mainloop()
    # main_thread.start()
    # root.continue_event.start()