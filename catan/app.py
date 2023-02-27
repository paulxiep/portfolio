from engine import *
from tkinter import *
from math import sqrt
import random

class CatanCanvas(Canvas):
    def __init__(self, master, coor=(200, 100), *args, **kwargs):
        Canvas.__init__(self, master=master, *args, **kwargs)
        self.coor = coor
        self.hex_color_code = {'ore': 'gray', 'brick': 'orange',
                           'wool': 'white', 'grain': 'yellow',
                           'lumber': 'green', 'desert': 'red',
                           'none': 'blue'}
        self.player_color_code = {None: 'black', **{item: item for item in ['orange red', 'blue', 'floral white']}}
    def draw_hex(self, coor, hex, v=40):
        x, y = coor
        cos60 = 1/2
        sin60 = sqrt(3)/2
        self.create_polygon([x, y, x+v*sin60, y-v*cos60,
                             x+2*v*sin60, y, x+2*v*sin60, y+v,
                             x+v*sin60, y+v+v*cos60, x, y+v], outline='black',
                            fill=self.hex_color_code[hex.resource], width=1)
        self.create_text(x+v*sin60, y+v*cos60, fill='black',
                         text=hex.number*(not hex.robber),
                         font=('Helvetica 15 bold'))
    def draw_corner(self, coor, corner):
        x, y = coor
        if corner.city is None:
            size = 5
            self.create_oval(x-size, y-size, x+size, y+size, fill=self.player_color_code[corner.settlement])
        else:
            size = 8
            self.create_oval(x-size, y-size, x+size, y+size, fill=self.player_color_code[corner.city])
    def draw_edge(self, coor1, coor2):
        self.create_line(coor1[0], coor1[1], coor2[0], coor2[1], fill='black', width=3)
    def draw_board(self, board, v=40):
        cos60 = 1/2
        sin60 = sqrt(3)/2
        h_length = board.h_length
        d_length = board.d_length
        for y in range(1, d_length*2):
            for x in range(1, h_length+1+d_length-1-abs(y-d_length)):
                self.draw_hex((self.coor[0] + 2*v*sin60*(x-1-(1+d_length-1-abs(y-d_length))/2),
                               self.coor[1] + v*1.5*(y-1)),
                              hex=board.hexes[y][x], v=v)

        for y in range(1, d_length * 4):
            if y % 2 == 1:
                for x in range(1, h_length * 2 + 1 + (d_length - 1 - abs(d_length * 2 - 1 - y + 1) // 2) * 2):
                    if x % 2 == 1:
                        a = self.coor[0] + 2 * v * sin60 * ((x+1)/2 - 1 - (1 + d_length - 1 - abs((y+1)/ 2 - d_length)) / 2)
                        b = self.coor[1] + v * 1.5 * ((y+1)/ 2 - 1)
                        self.draw_edge((a, b), (a+v*sin60, b-v*cos60))
                    else:
                        if y <= d_length*2:
                            a = self.coor[0] + 2 * v * sin60 * ((x+2)/2 - 1 - (1 + d_length - 1 - abs((y+1) / 2 - d_length)) / 2)
                            b = self.coor[1] + v * 1.5 * ((y+1) / 2 - 1)
                        else:
                            a = self.coor[0] + 2 * v * sin60 * ((x)/2 - 1 - (1 + d_length - 1 - abs((y+1) / 2 - d_length)) / 2)
                            b = self.coor[1] + v * 1.5 * ((y+1) / 2 - 1)
                        self.draw_edge((a, b), (a-v*sin60, b-v*cos60))
            else:
                for x in range(1, h_length + d_length + 1 - abs(d_length - y // 2 - 1 + 1)):
                    a = self.coor[0] + 2 * v * sin60 * (x - 1 - (1 + d_length - 1 - abs(y/2 - d_length)) / 2)
                    b = self.coor[1] + v * 1.5 * (y/2 - 1)
                    self.draw_edge((a, b), (a, b+v))

        for y in range(1, d_length * 4 + 1):
            for x in range(1, h_length + d_length + 1 - int(abs((d_length * 4 - 1) / 2 - y + 1) + 1) // 2):
                if y % 2 == 1:
                    if y <= d_length*2:
                        self.draw_corner((self.coor[0] + v*sin60 + 2*v*sin60*(x-1-(1+d_length-1-abs((y+1)/2-d_length))/2),
                                       self.coor[1] - v*cos60 + v*1.5*((y+1)/2-1)), corner=board.corners[y][x])
                    else:
                        self.draw_corner((self.coor[0] - v*sin60 + 2*v*sin60*(x-1-(1+d_length-1-abs((y+1)/2-d_length))/2),
                                       self.coor[1] - v*cos60 + v*1.5*((y+1)/2-1)), corner=board.corners[y][x])

                else:
                    self.draw_corner((self.coor[0] + 2 * v * sin60 * (
                                x - 1 - (1 + d_length - 1 - abs(y/2 - d_length)) / 2),
                                      self.coor[1] + v * 1.5 * (y/2 - 1)), corner=board.corners[y][x])



if __name__ == '__main__':
    root = Tk()
    root.title('Catan')
    root.geometry('800x600')
    catan_board = CatanHexBoard(6)
    # for corner in catan_board.corner_list:
    #     corner.city = random.choice(['orange red', 'blue', 'floral white', None])
    catan_canvas = CatanCanvas(root, width=600, height=600)
    catan_canvas.draw_board(catan_board)
    catan_canvas.place(x=0, y=0)
    root.mainloop()