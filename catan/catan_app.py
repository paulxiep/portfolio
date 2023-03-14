from tkinter import *
from catan_session import CatanSession
from catan_canvas import CatanCanvas
# from catan_units.catan_hex import *
from catan_units.catan_ai import CatanAI
from catan_units.ai_choices import sub_ais
from catan_units.game_parameters import resource_types, development_cards, build_options, die

import random
from functools import reduce, partial
from itertools import groupby

import logging
from gtts import gTTS
import playsound

logging.basicConfig(level=logging.INFO)

class CatanApp(CatanSession, Tk):
    def __init__(self, players, board_type=4, board=None, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        CatanSession.__init__(self, players, board_type, board)
        self.title('Catan')
        self.geometry('1100x600')
        self.canvas_frame = Frame(self, width=500, height=600)
        self.canvas = CatanCanvas(self.canvas_frame, self, dict([(player, color) for player, color, _, _ in players]), width=500, height=600)
        self.canvas.place(x=0, y=0)
        self.draw_board()
        self.canvas_frame.grid(row=0, column=0)
        self.ui_frame = Frame(self, width=600, height=600)
        self.ui_frame.grid(row=0, column=1)
        self.state_label = Label(self, text='Catan', font='Helvetica 18 bold')
        self.state_label.place(x=10, y=10)
        self.instruction_label = Label(self, text='Click Proceed', font='Helvetica 16')
        self.instruction_label.place(x=610, y=50)
        self.generic_ui_frame = Frame(self.ui_frame, width=600, height=100)
        self.turn_ui_frame = Frame(self.ui_frame, width=600, height=500)
        self.generic_ui_frame.grid(row=1, column=0)
        self.turn_ui_frame.grid(row=0, column=0)
        self.resource_data = {player: {} for player in self.players.keys()}
        self.resource_spin = {player: {} for player in self.players.keys()}
        self.player_columns = {}
        self.resource_rows = {}
        self.continue_var = BooleanVar(False)
        self.continue_button = Button(self.generic_ui_frame, text='Proceed', command=self.advance_state)
        self.continue_button.place(x=300, y=20)
        self.randomize_board_button = Button(self.generic_ui_frame, text='Randomize', command=self.randomize_board)
        self.randomize_board_button.place(x=200, y=20)
        self.quit_button = Button(self.generic_ui_frame, text='Exit', command=self.exit)
        self.quit_button.place(x=400, y=20)
        self.table = Frame(self.turn_ui_frame, width=300)
        self.table.place(x=20, y=150)
        self.build_buttons = {
            build: Button(self.turn_ui_frame, text=f'buy {build}', command=partial(lambda to_buy: self.buy(to_buy=to_buy, player=self.game_state[1]), to_buy=build),
                          state='disabled') for build in build_options.keys()}
        [build_button.place(x=20 + 100 * i, y=470) for i, build_button in enumerate(self.build_buttons.values())]
        Label(self.turn_ui_frame, text='trade away').place(x=260, y=170)
        Label(self.turn_ui_frame, text='trade for').place(x=400, y=170)
        self.trade_button = Button(self.turn_ui_frame, text='Trade', command=lambda:self.bank_trade(self.game_state[1]), state='disabled')
        self.trade_button.place(x=345, y=200)
        self.trade_away_var = StringVar(self)
        self.trade_away_option = OptionMenu(self.turn_ui_frame, self.trade_away_var, *resource_types)
        self.trade_away_option.place(x=260, y=200)
        self.trade_for_var = StringVar(self)
        self.trade_for_option = OptionMenu(self.turn_ui_frame, self.trade_for_var, *resource_types)
        self.trade_for_option.place(x=400, y=200)
        self.player_trade_var = StringVar(self)
        self.player_trade_option = OptionMenu(self.turn_ui_frame, self.player_trade_var, ())
        self.player_trade_option.place(x=260, y=240)
        self.player_trade_button = Button(self.turn_ui_frame, text='Player Trade', command=lambda:self.player_trade(self.game_state[1]), state='disabled')
        self.player_trade_button.place(x=260, y=280)
        self.longest_road_data = Label(self.turn_ui_frame, text='none')
        self.longest_road_data.place(x=360, y=320)
        self.largest_army_data = Label(self.turn_ui_frame, text='none')
        self.largest_army_data.place(x=360, y=350)
        Label(self.turn_ui_frame, text='longest_road:').place(x=260, y=320)
        Label(self.turn_ui_frame, text='largest_army:').place(x=260, y=350)
        self.development_var = StringVar(self)
        Label(self.turn_ui_frame, text='development').place(x=260, y=380)
        self.development_option = OptionMenu(self.turn_ui_frame, self.development_var,
                                             *[card for card in set(development_cards) if card != 'victory'])
        self.development_option.place(x=260, y=400)
        self.use_development_button = Button(self.turn_ui_frame, text='Use', command=lambda:self.use_development(self.game_state[1]))
        self.use_development_button.place(x=360, y=400)
        self.dice_var = IntVar(self)
        Label(self.turn_ui_frame, text='dice').place(x=260, y=440)
        self.dice_entry = Entry(self.turn_ui_frame, textvariable=self.dice_var, width=5)
        self.dice_entry.place(x=320, y=440)

        for j, resource in enumerate(list(self.players.values())[0].resource.keys()):
            self.resource_rows[resource] = Label(self.table, text=resource)
            self.resource_rows[resource].grid(row=j + 1, column=0)
        self.resource_rows['army'] = Label(self.table, text='army')
        self.resource_rows['army'].grid(row=j+2, column=0)
        for k, development in enumerate(list(self.players.values())[0].development.keys()):
            self.resource_rows[resource] = Label(self.table, text=development)
            self.resource_rows[resource].grid(row=j + 3 + k, column=0)
        Label(self.table, text='points').grid(row=14, column=0)
        self.point_data = {}
        for i, player in enumerate(self.players.keys()):
            self.point_data[player] = Label(self.table, text=0)
            self.point_data[player].grid(row=14, column=i + 1)
            self.player_columns[player] = Label(self.table, text=player)
            self.player_columns[player].grid(row=0, column=i + 1)
            for j, resource in enumerate(self.players[player].resource.keys()):
                self.resource_data[player][resource] = IntVar(self, value=0)
                self.resource_spin[player][resource] = Spinbox(self.table, values=tuple(range(20)), width=4,
                                                               textvariable=self.resource_data[player][resource],
                                                               state='normal')
                self.resource_spin[player][resource].grid(row=j + 1, column=i + 1)

            self.resource_spin[player]['army'] = Label(self.table, text=0)
            self.resource_spin[player]['army'].grid(row=j + 2, column=i + 1)
            for k, development in enumerate(self.players[player].development.keys()):
                self.resource_spin[player][development] = Label(self.table, text=0)
                self.resource_spin[player][development].grid(row=j + 3 + k, column=i + 1)

    def instruction(self, instruction):
        print(instruction)
        self.instruction_label.config(text=instruction)
        try:
            gTTS(text=instruction, lang='en').save('temp.mp3')
            playsound.playsound('.\\temp.mp3')
        except:
            pass

    def draw_board(self):
        self.canvas.draw_board(self.board)

    def randomize_board(self):
        self.board.assign_resources()
        self.board.assign_numbers()
        self.canvas.delete('all')
        self.draw_board()

    def exit(self):
        self.destroy()

    def resolve_state(self):
        if len(self.game_state) > 0:
            player = self.players[self.game_state[1]]
            for p in self.players.keys():
                for key in resource_types:
                    self.players[p].resource[key] = self.resource_data[p][key].get()
            if isinstance(player, CatanAI):
                if self.game_state[0] == 'setup_1':
                    if self.game_state[2]=='settlement':
                        coor = player.sub_ais['first_settlement_ai'](self, player)
                        self.place('settlement')(player.name, coor)
                        self.point_data[player.name].config(text=player.points)
                        self.update()
                elif self.game_state[0] == 'setup_2':
                    if self.game_state[2]=='settlement':
                        coor = player.sub_ais['second_settlement_ai'](self, player)
                        self.place('settlement')(player.name, coor)
                        self.point_data[player.name].config(text=player.points)
                        self.update()
                if self.game_state[2]=='road':
                    coor = player.sub_ais['initial_road_ai'](self, player)
                    self.place('road')(player.name, coor)
                elif self.game_state[2]=='trade':
                    options = player.sub_ais['player_trade_ai'](self, player)
                    while len(options)>0:
                        self.player_trade(player.name, options)
                        options = player.sub_ais['player_trade_ai'](self, player)

                    trades=player.sub_ais['bank_trade_ai'](self, player)
                    print(trades)
                    traded_away = []
                    while len(trades)>0:
                        for trade in trades:
                            if trade[0] not in traded_away:
                                self.bank_trade(self.game_state[1], trade[0], trade[1])
                                traded_away.append(trade[0])
                        trades = player.sub_ais['bank_trade_ai'](self, player)
                elif self.game_state[2]=='build':
                    builds = player.sub_ais['build_ai'](self, player)
                    print(builds)
                    while len(builds)>0:
                        for build in builds:
                            if self.buy(build[0], player.name):
                                self.place(build[0])(player.name, build[1])
                                if build[0] in ['settlement', 'city']:
                                    self.point_data[player.name].config(text=player.points)
                                    self.update()
                                self.canvas.delete('all')
                                self.draw_board()
                        builds = player.sub_ais['build_ai'](self, player)
                elif self.game_state[2] == 'dice':
                    if player.sub_ais['knight_ai'](self, player):
                        self.use_development(player.name, 'knight')
                elif self.game_state[2] == 'resolve_dice':
                    for item in player.sub_ais['development_ai'](self, player):
                        self.use_development(player.name, item)
                self.canvas.delete('all')
                self.draw_board()
                self.update()
            if len(self.pending_process) > 0:
                self.instruction('resolve current process before proceeding')
                return False
            if self.game_state[2] == 'dice':
                try:
                    self.dice = int(self.dice_var.get())
                    assert self.dice in range(1, 13)
                except:
                    self.dice = random.choice(die) + random.choice(die)
                self.dice_var.set(0)
                self.dice_entry.update()
                self.instruction(f'{self.game_state[1]} rolled {self.dice}')
                return True
        else:
            self.randomize_board_button.config(state='disabled')
        return True

    def advance_state(self):
        if self.resolve_state():
            self.game_state = next(self.game_state_generator)
            print(self.game_state)
            player = self.game_state[1]
            if self.game_state[2] == 'dice':
                while len(self.players[player].inactive_development) > 0:
                    development = self.players[player].inactive_development.pop()
                    self.players[player].development[development] += 1
                    self.resource_spin[player][development].config(text=self.players[player].development[development])
                    if development == 'victory':
                        self.players[player].points += 1
                        self.point_data[player].config(text=self.players[player].points)
            self.create_turn_buttons(self.game_state[1])
            self.state_label.config(text=self.game_state)
        else:
            self.instruction('resolve all actions before proceeding')

    def moving_robber(self, player):
        super().moving_robber(player)
        self.resource_spin[player]['knight'].config(text=self.players[player].development['knight'])
        self.resource_spin[player]['army'].config(text=self.players[player].army)
        if not isinstance(self.players[player], CatanAI):
            self.canvas.draw_robber_buttons(self.board, player)
        self.update()

    def check_largest_army(self, player):
        super().check_largest_army(player)
        largest_army = 'none'
        for p in self.players.values():
            self.point_data[p.name].config(text=p.points)
            if p.largest_army:
                largest_army = p.name
        self.largest_army_data.config(text=largest_army)
        self.update()

    def use_plenty(self, player, no_process=False):
        self.instruction(f'{player} take free resource')
        if not no_process:
            self.pending_process.append('plenty')
            self.pending_process.append('plenty')
            self.resource_spin[player]['plenty'].config(text=self.players[player].development['plenty'])
        self.canvas.draw_resource_buttons(self.board, player)
        self.update()
        if isinstance(self.players[player], CatanAI):
            self.pending_process.pop()
            resource1, resource2 = self.players[player].sub_ais['plenty_ai'](self, self.players[player])
            self.players[player].resource[resource2] += 1
            self.resource_data[player][resource2].set(self.players[player].resource[resource2])
            self.resource_choose(player, resource1)

    def use_monopoly(self, player):
        self.instruction(f'{player} choose resource to monopolize')
        self.pending_process.append('monopoly')
        self.canvas.draw_resource_buttons(self.board, player)
        self.resource_spin[player]['monopoly'].config(text=self.players[player].development['monopoly'])
        self.update()
        if isinstance(self.players[player], CatanAI):
            resource = self.players[player].sub_ais['monopoly_ai'](self, self.players[player])
            self.resource_choose(player, resource)

    def use_road(self, player):
        self.resource_spin[player]['road'].config(text=self.players[player].development['road'])
        self.update()

    def reset_player_trade_options(self, options):
        menu = self.player_trade_option["menu"]
        menu.delete(0, "end")
        for string in options:
            menu.add_command(label=string,
                             command=lambda value=string:
                             self.player_trade_var.set(value))
        self.player_trade_var.set('')

    def create_turn_buttons(self, current_player):
        if self.game_state[-1]=='road':
            self.pending_process.append('road')
            eligible = self.compile_eligible_road_initial(self.players[current_player])
            self.canvas.draw_road_placer(self.board, self, eligible=eligible)
            self.update()
        elif self.game_state[-1]=='settlement':
            self.pending_process.append('settlement')
            self.instruction(f'{current_player} place settlement')
            eligible = self.compile_eligible_settlement_initial()
            self.canvas.draw_buttons(self.board, self, eligible=eligible)
            self.update()
        elif self.game_state[-1] == 'collect_resource':
            to_collect = {resource: 0 for resource in resource_types}
            last_action = self.players[current_player].last_action
            for hex in self.board.corners[last_action[1]][last_action[0]].hexes.values():
                if hex is not None and hex.resource != 'desert':
                    to_collect[hex.resource] += 1
                    self.players[current_player].resource[hex.resource] += 1
                    self.resource_data[current_player][hex.resource].set(self.players[current_player].resource[hex.resource])
            to_collect_text = ' and '.join([f'{v} {k}' for k, v in to_collect.items() if v>0])
            self.instruction(f'{current_player} collects {to_collect_text}')
        elif self.game_state[-1] == 'dice':
            self.instruction(f'{current_player} throws dice')
        elif self.game_state[-1] == 'resolve_dice' and self.dice!=7:
            collect = {player: {resource: 0 for resource in resource_types} for player in self.players.keys()}
            for corner in self.board.corner_list:
                if corner.settlement is not None:
                    for hex in corner.hexes.values():
                        if hex is not None and hex.number == self.dice and not hex.robber:
                            collect[corner.settlement][hex.resource] += 1
                            self.players[corner.settlement].resource[hex.resource] += 1
                            self.resource_data[corner.settlement][hex.resource].set(self.players[corner.settlement].resource[hex.resource])

                if corner.city is not None:
                    for hex in corner.hexes.values():
                        if hex is not None and hex.number == self.dice and not hex.robber:
                            collect[corner.city][hex.resource] += 2
                            self.players[corner.city].resource[hex.resource] += 2
                            self.resource_data[corner.city][hex.resource].set(self.players[corner.city].resource[hex.resource])

            to_collect_text = ',\n'.join(list(filter(lambda x: not x.endswith('ects '), [f'{player} collects ' + ', '.join([f'{v} {k}' for k, v in collect[player].items() if v > 0]) for player in self.players.keys()])))
            if len(to_collect_text)==0:
                to_collect_text = 'Click Proceed'
            self.instruction(to_collect_text)
        elif self.game_state[-1] == 'resolve_dice' and self.dice==7:
            to_discard = {}
            for player in self.players.keys():
                total = reduce(list.__add__, [[resource] * self.players[player].resource[resource] for resource in resource_types])
                if len(total) > 7:
                    to_discard[player] = sorted(random.sample(total, len(total)//2))
                    for resource in to_discard[player]:
                        self.players[player].resource[resource] -= 1
                        self.resource_data[player][resource].set(self.players[player].resource[resource])
            to_discard_text = ',\n'.join(list(filter(lambda x: not x.endswith('ards '), [f'{player} discards ' + ', '.join([f'{v} {k}' for k, v in {key: len(list(res)) for key, res in groupby(to_discard[player])}.items()]) for player in to_discard.keys()])))
            if len(to_discard_text)==0:
                to_discard_text = 'Click proceed after moving robber'
            self.instruction(to_discard_text)
            self.moving_robber(current_player)
        else:
            self.instruction_label.config(text='Click Proceed')
        if self.game_state[-1]=='trade':
            trade_list = []
            for player in self.players.values():
                if isinstance(player, CatanAI):
                    player.sub_ais['strategy_ai'](self, player)
                if player.name != current_player and isinstance(player, CatanAI):
                    trade_list += player.sub_ais['player_trade_ai'](self, player, self.players[current_player])
            self.reset_player_trade_options(trade_list)
            self.trade_list = trade_list
        self.trade_button.config(state='disabled')
        self.player_trade_button.config(state='disabled')
        [build_button.config(state='disabled') for build_button in self.build_buttons.values()]
        if self.game_state[-1] == 'build':
            [build_button.config(state='normal') for build_button in self.build_buttons.values()]
        elif self.game_state[-1] == 'trade':
            self.trade_button.config(state='normal')
            self.player_trade_button.config(state='normal')

    def buy(self, to_buy, player, free=False):
        if super().buy(to_buy, player, free):
            for k, v in build_options[to_buy].items():
                self.resource_data[player][k].set(self.players[player].resource[k])
            if not isinstance(self.players[player], CatanAI):
                if to_buy == 'road':
                    # self.instruction(f'{player} place road')
                    self.canvas.draw_road_placer(self.board, self, eligible=self.compile_eligible_road(self.players[player]))
                    self.update()
                if to_buy == 'settlement':
                    # self.instruction(f'{player} place settlement')
                    self.canvas.draw_buttons(self.board, self, eligible=self.compile_eligible_settlement(self.players[player]))
                    self.update()
                if to_buy == 'city':
                    # self.instruction(f'{player} place city')
                    self.canvas.draw_buttons(self.board, self, eligible=self.compile_eligible_city(self.players[player]))
                    self.update()
            self.instruction(f'{player} purchased {to_buy}')
            return True

    def player_trade(self, current_player, options=None):
        def perform_trade(player, trade_away, trade_for):
            self.players[current_player].resource[trade_away] -= 1
            self.players[player].resource[trade_away] += 1
            self.players[current_player].resource[trade_for] += 1
            self.players[player].resource[trade_for] -= 1
            self.resource_data[player][trade_away].set(self.players[player].resource[trade_away])
            self.resource_data[player][trade_for].set(self.players[player].resource[trade_for])
            self.resource_data[current_player][trade_away].set(self.players[current_player].resource[trade_away])
            self.resource_data[current_player][trade_for].set(self.players[current_player].resource[trade_for])

        if options is None:
            if self.player_trade_var.get() == '':
                print('no player trade option chosen')
                return None
            else:
                player, trade = tuple(self.player_trade_var.get().split(': '))
                trade_away, trade_for = tuple(trade.split(' for '))
                perform_trade(player, trade_away, trade_for)
                self.instruction(f"{current_player} exchanged {trade_away} for {player}'s {trade_for}")
        else:
            for player, trade_away, trade_for in options:
                perform_trade(player, trade_away, trade_for)
                print(f"{current_player} exchanged {trade_away} for {player}'s {trade_for}")
        new_options = []
        for player in self.players.keys():
            if player != current_player and isinstance(player, CatanAI):
                new_options += self.players[player].sub_ais['player_trade_ai'](self, self.players[player], self.players[current_player])
        self.trade_list = new_options
        self.reset_player_trade_options(new_options)

    def bank_trade(self, player, trade_away=None, trade_for=None):
        if super().bank_trade(player, trade_away, trade_for):
            if trade_away is None and trade_for is None:
                trade_away = self.trade_away_var.get()
                trade_for = self.trade_for_var.get()
            self.resource_data[player][trade_away].set(self.players[player].resource[trade_away])
            self.resource_data[player][trade_for].set(self.players[player].resource[trade_for])

    def corner_place(self, player, coor):
        i, j = coor
        if self.board.corners[j][i].settlement is None:
            self.place('settlement')(player, coor)
        else:
            self.place('city')(player, coor)
        self.point_data[player].config(text=self.players[player].points)
        self.update()
        self.players[player].last_action = coor
        self.canvas.delete('all')
        self.draw_board()

    def road_place(self, player, coor):
        self.place('road')(player, coor)
        self.players[player].last_action = coor
        self.canvas.delete('all')
        self.draw_board()
        if self.road:
            self.road = False
            self.buy('road', player, free=True)

    def robber_move(self, player, coor, robber_coor):
        to_steal = super().robber_move(player, coor, robber_coor)
        if to_steal is not None:
            other, resource = to_steal
            self.resource_data[player][resource].set(self.players[player].resource[resource])
            self.resource_data[other][resource].set(self.players[other].resource[resource])
        self.pending_process.pop(0)
        self.canvas.delete('all')
        self.draw_board()
        self.update()

    def resource_choose(self, current_player, resource):
        if self.plenty:
            self.players[current_player].resource[resource] += 1
            self.resource_data[current_player][resource].set(self.players[current_player].resource[resource])
            if len(self.pending_process) > 1:
                self.pending_process.pop(0)
                self.use_plenty(current_player, True)
            else:
                self.pending_process.pop(0)
                self.plenty=False
                self.canvas.delete('all')
                self.draw_board()
        elif self.monopoly:
            take_text = []
            for player in self.players.keys():
                if player != current_player:
                    resource_count = self.players[player].resource[resource]
                    if resource_count > 0:
                        self.players[player].resource[resource] = 0
                        self.resource_data[player][resource].set(self.players[player].resource[resource])
                        self.players[current_player].resource[resource] += resource_count
                        self.resource_data[current_player][resource].set(self.players[current_player].resource[resource])
                        take_text += [f'{current_player} takes {resource_count} {resource} from {player}']
            self.monopoly = False
            self.pending_process.pop(0)
            if len(take_text)>0:
                self.instruction(',\n'.join(take_text))
            self.canvas.delete('all')
            self.draw_board()
        self.update()

if __name__ == '__main__':
    player = ['blue', 'red', 'white', 'orange']
    color = ['blue', 'red', 'white', 'orange']
    ai = [True, True, True, False]
    personality = {sub_ai: 'basis' for sub_ai in sub_ais}
    personalities = [personality, personality, personality, personality]
    root = CatanApp(list(zip(player, color, ai, personalities)))
    root.mainloop()