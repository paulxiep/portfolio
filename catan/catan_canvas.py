from tkinter import *
from PIL import ImageTk, ImageDraw, Image
from math import sqrt

class CatanCanvas(Canvas):
    def __init__(self, master, session, coor=(200, 100), *args, **kwargs):
        Canvas.__init__(self, master=master, *args, **kwargs)
        self.coor = coor
        self.hex_color_code = {'ore': 'gray', 'brick': 'orange red',
                           'wool': 'beige', 'grain': 'yellow',
                           'lumber': 'green', 'desert': 'red',
                           'none': 'light sky blue'}
        self.player_color_code = {None: 'black', **{item: item for item in ['orange', 'blue', 'white']}}
        self.corner_button_image = ImageTk.PhotoImage(self.create_corner_button_img())
        self.session = session

    def create_corner_button_img(self):
        img = Image.new('RGBA', (10, 10), (0, 0, 0, 255))

        draw = ImageDraw.Draw(img)
        draw.ellipse((2, 2, 8, 8), fill=(55, 55, 55))
        return img
    def draw_resource_buttons(self, board, v=40):
        cos60 = 1 / 2
        sin60 = sqrt(3) / 2
        h_length = board.h_length
        d_length = board.d_length
        self.resource_buttons = {y: {x: None for x, hex in row.items()} for y, row in board.hexes.items()}
        for y in range(1, d_length * 2):
            for x in range(1, h_length + 1 + d_length - 1 - abs(y - d_length)):
                if board.hexes[y][x].resource != 'desert':
                    self.draw_resource_button((self.coor[0] + 2 * v * sin60 * (x - 1 - (1 + d_length - 1 - abs(y - d_length)) / 2),
                                   self.coor[1] + v * 1.5 * (y - 1)),
                                  board.hexes[y][x], v, x, y)

                    self.tag_bind(self.resource_buttons[y][x], f'<Button-1>', self.choose_resource(board.hexes[y][x].resource))
    def draw_resource_button(self, coor, hex, v, i, j):
        x, y = coor
        cos60 = 1/2
        sin60 = sqrt(3)/2
        x, y = coor
        cos60 = 1/2
        sin60 = sqrt(3)/2
        self.resource_buttons[j][i] = self.create_text(x+v*sin60, y+v*cos60, fill='dodger blue',
                         text=hex.resource,
                         font=('Helvetica 15 bold'))

    def draw_robber_button(self, coor, hex, v, i, j):
        x, y = coor
        cos60 = 1/2
        sin60 = sqrt(3)/2
        self.robber_buttons[j][i] = self.create_text(x+v*sin60, y+v*cos60, fill='black'*(not hex.robber) + 'sky blue'*hex.robber,
                         text=hex.number,
                         font=('Helvetica 15 bold'))
    def draw_hex(self, coor, hex, v, i, j):
        x, y = coor
        cos60 = 1/2
        sin60 = sqrt(3)/2
        self.create_polygon([x, y, x+v*sin60, y-v*cos60,
                             x+2*v*sin60, y, x+2*v*sin60, y+v,
                             x+v*sin60, y+v+v*cos60, x, y+v], outline='black',
                            fill=self.hex_color_code[hex.resource], width=1, tags=['hex'])
        self.create_text(x+v*sin60, y+v*cos60, fill='black'*(not hex.robber) + 'sky blue'*hex.robber,
                         text=hex.number,
                         font=('Helvetica 15'), tags=['hex', 'number'])
    def draw_corner(self, coor, corner, i, j):
        x, y = coor
        if corner.city is None:
            size = 5
            self.create_oval(x-size, y-size, x+size, y+size, fill=self.player_color_code[corner.settlement], tags=['corner'])
        else:
            size = 10
            self.create_oval(x-size, y-size, x+size, y+size, fill=self.player_color_code[corner.city], tags=['corner'])
        if corner.harbor is not None:
            # print(corner, i, j, 'has harbor', corner.harbor)
            self.create_text(x, y, fill='magenta', text=corner.harbor[0].upper(), font=('Helvetica 18 bold'))
    def draw_edge(self, coor1, coor2, edge, i, j):
        self.create_line(coor1[0], coor1[1], coor2[0], coor2[1], fill=self.player_color_code[edge.road], width=5, tags=['edge'])
    def draw_edge_button(self, coor1, coor2, i, j):
        self.edges[j][i] = self.create_line(coor1[0], coor1[1], coor2[0], coor2[1], fill='grey', width=5, tags=['edge'])
    def draw_robber_buttons(self, board, session, v=40):
        cos60 = 1 / 2
        sin60 = sqrt(3) / 2
        h_length = board.h_length
        d_length = board.d_length
        self.robber_buttons = {y: {x: None for x, hex in row.items()} for y, row in board.hexes.items()}
        for y in range(1, d_length * 2):
            for x in range(1, h_length + 1 + d_length - 1 - abs(y - d_length)):
                if board.hexes[y][x].robber:
                    a, b = x, y
        for y in range(1, d_length * 2):
            for x in range(1, h_length + 1 + d_length - 1 - abs(y - d_length)):
                if not board.hexes[y][x].robber and board.hexes[y][x].resource != 'desert':
                    self.draw_robber_button((self.coor[0] + 2 * v * sin60 * (x - 1 - (1 + d_length - 1 - abs(y - d_length)) / 2),
                                   self.coor[1] + v * 1.5 * (y - 1)),
                                  board.hexes[y][x], v, x, y)

                    self.tag_bind(self.robber_buttons[y][x], f'<Button-1>', self.robber_move(x, y, a, b))
    def draw_board(self, board, v=40):
        cos60 = 1/2
        sin60 = sqrt(3)/2
        h_length = board.h_length
        d_length = board.d_length
        for y in range(1, d_length*2):
            for x in range(1, h_length+1+d_length-1-abs(y-d_length)):
                self.draw_hex((self.coor[0] + 2*v*sin60*(x-1-(1+d_length-1-abs(y-d_length))/2),
                               self.coor[1] + v*1.5*(y-1)),
                              board.hexes[y][x], v, x, y)

        for y in range(1, d_length * 4):
            if y % 2 == 1:
                for x in range(1, h_length * 2 + 1 + (d_length - 1 - abs(d_length * 2 - 1 - y + 1) // 2) * 2):
                    if x % 2 == 1:
                        if y <= d_length*2:
                            a = self.coor[0] + 2 * v * sin60 * ((x+1)/2 - 1 - (1 + d_length - 1 - abs((y+1)/ 2 - d_length)) / 2)
                            b = self.coor[1] + v * 1.5 * ((y+1)/ 2 - 1)
                            self.draw_edge((a, b), (a + v * sin60, b - v * cos60), board.edges[y][x], x, y)
                        else:
                            a = self.coor[0] + 2 * v * sin60 * ((x+1)/2 - 1 - (1 + d_length - 1 - abs((y+1)/ 2 - d_length)) / 2)
                            b = self.coor[1] + v * 1.5 * ((y+1)/ 2 - 1)
                            self.draw_edge((a, b), (a-v*sin60, b-v*cos60), board.edges[y][x], x, y)
                    else:
                        if y <= d_length*2:
                            a = self.coor[0] + 2 * v * sin60 * ((x+2)/2 - 1 - (1 + d_length - 1 - abs((y+1) / 2 - d_length)) / 2)
                            b = self.coor[1] + v * 1.5 * ((y+1) / 2 - 1)
                            self.draw_edge((a, b), (a - v * sin60, b - v * cos60), board.edges[y][x], x, y)
                        else:
                            a = self.coor[0] + 2 * v * sin60 * ((x)/2 - 1 - (1 + d_length - 1 - abs((y+1) / 2 - d_length)) / 2)
                            b = self.coor[1] + v * 1.5 * ((y+1) / 2 - 1)
                            self.draw_edge((a, b), (a+v*sin60, b-v*cos60), board.edges[y][x], x, y)
            else:
                for x in range(1, h_length + d_length + 1 - abs(d_length - y // 2 - 1 + 1)):
                    a = self.coor[0] + 2 * v * sin60 * (x - 1 - (1 + d_length - 1 - abs(y/2 - d_length)) / 2)
                    b = self.coor[1] + v * 1.5 * (y/2 - 1)
                    self.draw_edge((a, b), (a, b+v), board.edges[y][x], x, y)

        for y in range(1, d_length * 4 + 1):
            for x in range(1, h_length + d_length + 1 - int(abs((d_length * 4 - 1) / 2 - y + 1) + 1) // 2):
                if y % 2 == 1:
                    if y <= d_length*2:
                        self.draw_corner((self.coor[0] + v*sin60 + 2*v*sin60*(x-1-(1+d_length-1-abs((y+1)/2-d_length))/2),
                                       self.coor[1] - v*cos60 + v*1.5*((y+1)/2-1)), board.corners[y][x], x, y)
                    else:
                        self.draw_corner((self.coor[0] - v*sin60 + 2*v*sin60*(x-1-(1+d_length-1-abs((y+1)/2-d_length))/2),
                                       self.coor[1] - v*cos60 + v*1.5*((y+1)/2-1)), board.corners[y][x], x, y)

                else:
                    self.draw_corner((self.coor[0] + 2 * v * sin60 * (
                                x - 1 - (1 + d_length - 1 - abs(y/2 - d_length)) / 2),
                                      self.coor[1] + v * 1.5 * (y/2 - 1)), board.corners[y][x], x, y)
    def draw_buttons(self, board, session, eligible=None, ineligible=None, v=40):
        # print('draw_buttons called')
        cos60 = 1/2
        sin60 = sqrt(3)/2
        h_length = board.h_length
        d_length = board.d_length
        print(eligible)
        self.buttons = {y: {x: None for x, corner in row.items()} for y, row in board.corners.items()}
        for y in range(1, d_length * 4 + 1):
            for x in range(1, h_length + d_length + 1 - int(abs((d_length * 4 - 1) / 2 - y + 1) + 1) // 2):
                if (eligible is not None and (x, y) in eligible) or (ineligible is not None and (x, y) not in ineligible):
                    if y % 2 == 1:
                        if y <= d_length*2:
                            self.buttons[y][x]=self.create_image(self.coor[0] + v*sin60 + 2*v*sin60*(x-1-(1+d_length-1-abs((y+1)/2-d_length))/2),
                                           self.coor[1] - v*cos60 + v*1.5*((y+1)/2-1), image=self.corner_button_image)
                        else:
                            self.buttons[y][x]=self.create_image(self.coor[0] - v*sin60 + 2*v*sin60*(x-1-(1+d_length-1-abs((y+1)/2-d_length))/2),
                                           self.coor[1] - v*cos60 + v*1.5*((y+1)/2-1), image=self.corner_button_image)

                    else:
                        self.buttons[y][x]=self.create_image(self.coor[0] + 2 * v * sin60 * (
                                    x - 1 - (1 + d_length - 1 - abs(y/2 - d_length)) / 2),
                                          self.coor[1] + v * 1.5 * (y/2 - 1), image=self.corner_button_image)
                    self.tag_bind(self.buttons[y][x], f'<Button-1>', self.corner_place(x, y))

    def draw_road_placer(self, board, session, eligible=None, ineligible=None, v=40):
        print('draw_road_placer called')
        cos60 = 1/2
        sin60 = sqrt(3)/2
        h_length = board.h_length
        d_length = board.d_length
        self.edges = {y: {x: None for x, edge in row.items()} for y, row in board.edges.items()}
        for y in range(1, d_length * 4):
            if y % 2 == 1:
                for x in range(1, h_length * 2 + 1 + (d_length - 1 - abs(d_length * 2 - 1 - y + 1) // 2) * 2):
                    if (eligible is not None and (x, y) in eligible) or (ineligible is not None and (x, y) not in ineligible):
                        if x % 2 == 1:
                            if y <= d_length*2:
                                a = self.coor[0] + 2 * v * sin60 * ((x+1)/2 - 1 - (1 + d_length - 1 - abs((y+1)/ 2 - d_length)) / 2)
                                b = self.coor[1] + v * 1.5 * ((y+1)/ 2 - 1)
                                self.draw_edge_button((a, b), (a + v * sin60, b - v * cos60), x, y)
                            else:
                                a = self.coor[0] + 2 * v * sin60 * ((x+1)/2 - 1 - (1 + d_length - 1 - abs((y+1)/ 2 - d_length)) / 2)
                                b = self.coor[1] + v * 1.5 * ((y+1)/ 2 - 1)
                                self.draw_edge_button((a, b), (a-v*sin60, b-v*cos60), x, y)
                        else:
                            if y <= d_length*2:
                                a = self.coor[0] + 2 * v * sin60 * ((x+2)/2 - 1 - (1 + d_length - 1 - abs((y+1) / 2 - d_length)) / 2)
                                b = self.coor[1] + v * 1.5 * ((y+1) / 2 - 1)
                                self.draw_edge_button((a, b), (a - v * sin60, b - v * cos60), x, y)
                            else:
                                a = self.coor[0] + 2 * v * sin60 * ((x)/2 - 1 - (1 + d_length - 1 - abs((y+1) / 2 - d_length)) / 2)
                                b = self.coor[1] + v * 1.5 * ((y+1) / 2 - 1)
                                self.draw_edge_button((a, b), (a+v*sin60, b-v*cos60), x, y)
                            # self.tag_bind(self.edges[y][x], f'<Button-1>', road_place(x, y, session))
                        self.tag_bind(self.edges[y][x], f'<Button-1>', self.road_place(x, y))
            else:
                for x in range(1, h_length + d_length + 1 - abs(d_length - y // 2 - 1 + 1)):
                    if (eligible is not None and (x, y) in eligible) or (ineligible is not None and (x, y) not in ineligible):
                        a = self.coor[0] + 2 * v * sin60 * (x - 1 - (1 + d_length - 1 - abs(y/2 - d_length)) / 2)
                        b = self.coor[1] + v * 1.5 * (y/2 - 1)
                        self.draw_edge_button((a, b), (a, b+v), x, y)
                        self.tag_bind(self.edges[y][x], f'<Button-1>', self.road_place(x, y))
    def corner_place(self, x, y):
        def place(event):
            self.session.corner_place(x, y)
        return lambda event: place(event)
    def road_place(self, x, y):
        def place(event):
            self.session.road_place(x, y)
        return lambda event: place(event)
    def robber_move(self, x, y, a, b):
        def move(event):
            self.session.move_robber(x, y, a, b)
        return lambda event: move(event)
    def choose_resource(self, resource):
        def move(event):
            self.session.choose_resource(resource)
        return lambda event: move(event)