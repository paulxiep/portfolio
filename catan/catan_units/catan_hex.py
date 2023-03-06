from functools import reduce
from random import randrange
from .hex_geometry import *

class CatanEdge(Edge):
    def __init__(self, coor=None, d_length=3):
        super().__init__(coor, d_length)
        self.road = None

    def __repr__(self):
        # return str(self.coor)
        return f'edge-{self.coor}-{self.road}'
        # if self.road:
        #     return 'R'
        # else:
        #     return 'X'


class CatanCorner(Corner):
    def __init__(self, coor=None):
        super().__init__(coor)
        self.settlement = None
        self.city = None
        self.harbor = None

    def hex_pips(self, number):
        return (6 - abs(7 - number)) * (number != 0)

    def resource_pips(self):
        out = {resource: 0 for resource in ['ore', 'brick', 'wool', 'grain', 'lumber']}
        for elem in self.hexes.values():
            if elem is not None and elem.number > 0:
                out[elem.resource] += self.hex_pips(elem.number)
        return out

    def pips(self):
        return reduce(int.__add__, [self.hex_pips(elem.number) for elem in list(self.hexes.values()) if elem is not None],
                   0)

    def pips_plus(self):
        return self.pips() + 2 * (self.harbor is not None)

    def __repr__(self):
        return f'corner-{self.coor}-{self.settlement}-{self.city}-{self.harbor}'
        # number = '-pips-' + str(self.pips())
        # if self.settlement:
        #     return str(self.settlement) + number
        # elif self.city:
        #     return str(self.city) + number
        # else:
        #     return number


class CatanHex(Hex):
    def __init__(self, coor=None):
        super().__init__(coor)
        self.resource = 'none'
        self.number = 0
        self.robber = False

    def pips(self):
        return (6 - abs(7 - self.number)) * (self.number != 0)


    def __repr__(self):
        return f'hex-{self.coor}-{self.resource}-{self.number}' + '-robber' * self.robber
        # return str(self.coor)
        # if not self.robber:
        #     return self.resource[0].upper() + str(self.number).zfill(2)
        # else:
        #     return self.resource[0].upper() + '00'


class CatanHexBoard(HexBoard):
    def __init__(self, max_players=4):
        self.resource_types = ['ore', 'brick', 'wool', 'grain', 'lumber', 'desert']
        if max_players == 4:
            super().__init__(3, 3)
            self.resource_count = dict(zip(self.resource_types, [3, 3, 4, 4, 4, 1]))
            self.numbers = reduce(list.__add__,
                                  map(lambda x: [x, x] if abs(x - 7) < 5 else [x], list(range(2, 7)) + list(range(8, 13))))
            self.harbor_list = [((1, 2), (1, 1)), ((2, 1), (3, 2)), ((4, 3), (5, 4)),
                                ((6, 6), (6, 7)), ((5, 9), (4, 10)), ((3, 11), (2, 12)),
                                ((1, 12), (1, 11)), ((1, 9), (1, 8)), ((1, 5), (1, 4))]
            self.harbor_types = ['x'] + ['ore', 'brick', 'wool', 'grain', 'lumber']
            for i, harbor in enumerate(self.harbor_list):
                if i % 3 == 0:
                    harbor_type = 'x'
                else:
                    harbor_type = self.harbor_types.pop(randrange(len(self.harbor_types)))
                for port in harbor:
                    self.corners[port[1]][port[0]].harbor = harbor_type
        elif max_players == 6:
            super().__init__(3, 4)
            self.resource_count = dict(zip(self.resource_types, [5, 5, 6, 6, 6, 2]))
            self.numbers = reduce(list.__add__,
                                  map(lambda x: [x, x, x] if abs(x - 7) < 5 else [x, x], list(range(2, 7)) + list(range(8, 13))))
        # self.players = ['orange', 'blue', 'white']
        self.resources = reduce(list.__add__, [[resource] * repeat for resource, repeat in self.resource_count.items()])
        self.robber = None
        # self.assign_resources()
        # self.assign_numbers()

    def assign_resources(self):
        resources = self.resources.copy()
        for h in self.hex_list:
            h.resource = resources.pop(randrange(len(resources)))

    def assign_numbers(self):
        def insert_numbers():
            numbers = self.numbers.copy()
            for h in self.hex_list:
                if h.resource != 'desert':
                    h.number = numbers.pop(randrange(len(numbers)))
                else:
                    h.robber = True
                    self.robber = h.coor

        def validate_board():
            for corner in self.corner_list:
                red = False
                for h in corner.hexes.values():
                    if h is not None and (h.number == 8 or h.number == 6):
                        if red:
                            return False
                        else:
                            red = True
            return True

        insert_numbers()
        while not validate_board():
            print('board invalidated')
            insert_numbers()

    def generate_elements(self):
        h_length = self.h_length
        d_length = self.d_length
        self.hexes = dict(
            map(lambda x: (x, {y: CatanHex((y, x)) for y in range(h_length + d_length - 1 - abs(d_length - x - 1))}),
                range(2 * d_length - 1)))
        self.edges = dict(map(lambda x: (
            x,
            {y: CatanEdge((y, x), self.d_length) for y in range(h_length * 2 + (d_length - 1 - abs(d_length * 2 - 1 - x) // 2) * 2)}),
                              range(d_length * 4 - 1)))
        self.corners = dict(map(lambda x: (x, {y: CatanCorner((y, x)) for y in range(
            h_length + d_length - int(abs((d_length * 4 - 1) / 2 - x) + 1) // 2)}), range(d_length * 4)))
