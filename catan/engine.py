from functools import reduce
from random import randrange


class Geometry:
    def __init__(self, coor=None):
        self.coor = coor

    def __repr__(self):
        return str(self.coor)

    def shift_coor(self):
        self.coor = (self.coor[0] + 1, self.coor[1] + 1)
        return self


class Hex(Geometry):
    def __init__(self, coor=None):
        super().__init__(coor)
        self.edges = {i * 2 - 1: None for i in range(1, 7)}
        self.corners = {i * 2: None for i in range(1, 7)}


class Edge(Geometry):
    def __init__(self, coor=None):
        super().__init__(coor)
        if coor[1] % 2 == 0:
            self.type = '|'
        else:
            if coor[0] % 2 == 1:
                self.type = '/'
            else:
                self.type = '\\'
        self.hexes = {'left': None, 'right': None}

    def __repr__(self):
        return self.type + str(self.coor)


class Corner(Geometry):
    def __init__(self, coor=None):
        super().__init__(coor)
        if coor[1] % 2 == 0:
            self.type = 'top'
            self.hexes = {4 * i - 2: None for i in range(1, 4)}
        else:
            self.type = 'bottom'
            self.hexes = {4 * i: None for i in range(1, 4)}


class CatanEdge(Edge):
    def __init__(self, coor=None):
        super().__init__(coor)
        self.road = None

    def __repr__(self):
        if self.road:
            return 'R00'
        else:
            return '100'


class CatanCorner(Corner):
    def __init__(self, coor=None):
        super().__init__(coor)
        self.settlement = None
        self.city = None

    def hex_pips(self, number):
        return (6 - abs(7 - number)) * (number != 0)

    def resource_pips(self):
        out = {resource: 0 for resource in ['ore', 'brick', 'wool', 'grain', 'lumber']}
        for elem in self.hexes.values():
            if elem is not None and elem.number > 0:
                out[elem.resource] += self.hex_pips(elem.number)
        return out

    def pips(self):
        return str(
            reduce(int.__add__, [self.hex_pips(elem.number) for elem in list(self.hexes.values()) if elem is not None],
                   0))

    def __repr__(self):
        number = str(self.pips())
        if self.settlement:
            return str(self.settlement) + number
        elif self.city:
            return str(self.city) + number
        else:
            return number


class CatanHex(Hex):
    def __init__(self, coor=None):
        super().__init__(coor)
        self.resource = 'none'
        self.number = 0
        self.robber = False

    def __repr__(self):
        if not self.robber:
            return self.resource[0].upper() + str(self.number).zfill(2)
        else:
            return self.resource[0].upper() + '00'


class HexBoard:
    def __init__(self, h_length, d_length):
        self.h_length = h_length
        self.d_length = d_length
        self.generate_elements()
        self.shift_coors()
        for j in self.edges.keys():
            if j % 2 == 0:
                self.edges[j] = {k: self.edges[j][k] for k in
                                 list(self.hexes[j // 2].keys()) + [len(self.hexes[j // 2].keys()) + 1]}
        self.assign_adjacent()
        self.hex_list = reduce(list.__add__, [list(v.values()) for v in self.hexes.values()])
        self.edge_list = reduce(list.__add__, [list(v.values()) for v in self.edges.values()])
        self.corner_list = reduce(list.__add__, [list(v.values()) for v in self.corners.values()])

    def generate_elements(self):
        h_length = self.h_length
        d_length = self.d_length
        self.hexes = dict(
            map(lambda x: (x, {y: Hex((y, x)) for y in range(h_length + d_length - 1 - abs(d_length - x - 1))}),
                range(2 * d_length - 1)))
        self.edges = dict(map(lambda x: (
            x, {y: Edge((y, x)) for y in range(h_length * 2 + (d_length - 1 - abs(d_length * 2 - 1 - x) // 2) * 2)}),
                              range(d_length * 4 - 1)))
        self.corners = dict(map(lambda x: (
            x, {y: Corner((y, x)) for y in range(h_length + d_length - int(abs((d_length * 4 - 1) / 2 - x) + 1) // 2)}),
                                range(d_length * 4)))

    def shift_coors(self):
        self.hexes = {k + 1: (lambda x: {kk + 1: vv.shift_coor() for kk, vv in v.items()})(v) for k, v in
                      self.hexes.items()}
        self.edges = {k + 1: (lambda x: {kk + 1: vv.shift_coor() for kk, vv in v.items()})(v) for k, v in
                      self.edges.items()}
        self.corners = {k + 1: (lambda x: {kk + 1: vv.shift_coor() for kk, vv in v.items()})(v) for k, v in
                        self.corners.items()}

    def assign_edges_to_hexes(self):
        h_length = self.h_length
        d_length = self.d_length
        for j in range(1, 2 * d_length):
            for i in range(1, h_length + d_length - abs(d_length - j - 1 + 1)):
                #                 print(j, i)
                self.hexes[j][i].edges[11] = self.edges[j * 2 - 1][2 * i - 1]
                self.hexes[j][i].edges[1] = self.edges[j * 2 - 1][2 * i]
                self.hexes[j][i].edges[3] = self.edges[j * 2][i + 1]
                self.hexes[j][i].edges[9] = self.edges[j * 2][i]
                self.hexes[j][i].edges[5] = self.edges[j * 2 + 1][2 * i - 1]
                self.hexes[j][i].edges[7] = self.edges[j * 2 + 1][2 * i]

    def assign_corners_to_hexes(self):
        h_length = self.h_length
        d_length = self.d_length
        for j in range(1, 2 * d_length):
            for i in range(1, h_length + d_length - abs(d_length - j - 1 + 1)):
                #                 print(j, i)
                if j < d_length:
                    self.hexes[j][i].corners[12] = self.corners[j * 2 - 1][i]
                    self.hexes[j][i].corners[10] = self.corners[j * 2][i]
                    self.hexes[j][i].corners[2] = self.corners[j * 2][i + 1]
                    self.hexes[j][i].corners[8] = self.corners[j * 2 + 1][i]
                    self.hexes[j][i].corners[4] = self.corners[j * 2 + 1][i + 1]
                    self.hexes[j][i].corners[6] = self.corners[j * 2 + 2][i + 1]
                else:
                    self.hexes[j][i].corners[12] = self.corners[j * 2 - 1][i]
                    self.hexes[j][i].corners[10] = self.corners[j * 2][i]
                    self.hexes[j][i].corners[2] = self.corners[j * 2][i + 1]
                    self.hexes[j][i].corners[8] = self.corners[j * 2 + 1][i]
                    self.hexes[j][i].corners[4] = self.corners[j * 2 + 1][i + 1]
                    self.hexes[j][i].corners[6] = self.corners[j * 2 + 2][i]

    def assign_hexes_to_edges(self):
        h_length = self.h_length
        d_length = self.d_length
        for j in range(1, d_length * 4):
            if j % 2 == 1:
                for i in range(1, h_length * 2 + 1 + (d_length - 1 - abs(d_length * 2 - 1 - j + 1) // 2) * 2):
                    if self.edges[j][i].type == '/':
                        if j > 1:
                            if i > 1:
                                self.edges[j][i].hexes['left'] = self.hexes[j // 2][i // 2]
                        if j < d_length * 4 - 1:
                            if i < h_length * 2 + (d_length - 1 - abs(d_length * 2 - 1 - j + 1) // 2) * 2:
                                self.edges[j][i].hexes['right'] = self.hexes[(j + 1) // 2][(i + 1) // 2]
                    if self.edges[j][i].type == '\\':
                        if j > 1:
                            if i < h_length * 2 + (d_length - 1 - abs(d_length * 2 - 1 - j + 1) // 2) * 2:
                                self.edges[j][i].hexes['right'] = self.hexes[j // 2][i // 2]
                        if j < d_length * 4 - 1:
                            if i > 1:
                                self.edges[j][i].hexes['left'] = self.hexes[(j + 1) // 2][(i) // 2]
            else:
                for i in range(1, h_length + d_length - abs(d_length - j // 2 - 1 + 1)):
                    if i > 1:
                        self.edges[j][i].hexes['left'] = self.hexes[j // 2][i - 1]
                    if i < h_length + d_length - abs(d_length - j - 1 + 1) - 1:
                        self.edges[j][i].hexes['right'] = self.hexes[j // 2][i]

    def assign_hexes_to_corners(self):
        h_length = self.h_length
        d_length = self.d_length
        for j in range(1, d_length * 4 + 1):
            for i in range(1, h_length + d_length + 1 - int(abs((d_length * 4 - 1) / 2 - j + 1) + 1) // 2):
                if self.corners[j][i].type == 'top':
                    #                     print(h_length + d_length - int(abs((d_length*4-1)/2 - j+1)+1)//2)
                    if j < d_length * 2:
                        self.corners[j][i].hexes[6] = self.hexes[(j + 1) // 2][i]
                    elif d_length * 2 < j < d_length * 4 - 1 and i > 1 and i < h_length + d_length - int(
                            abs((d_length * 4 - 1) / 2 - j + 1) + 1) // 2:
                        self.corners[j][i].hexes[6] = self.hexes[(j + 1) // 2][i - 1]
                    if 1 < j:
                        if i > 1:
                            self.corners[j][i].hexes[10] = self.hexes[j // 2][i - 1]
                        if i < h_length + d_length - int(abs((d_length * 4 - 1) / 2 - j + 1) + 1) // 2:
                            self.corners[j][i].hexes[2] = self.hexes[j // 2][i]
                else:
                    if j > 2:
                        if i > 1 and i < h_length + d_length - int(abs((d_length * 4 - 1) / 2 - j + 1) + 1) // 2:
                            self.corners[j][i].hexes[12] = self.hexes[(j - 2) // 2][i - 1]
                    if j < d_length * 4:
                        if i > 1:
                            self.corners[j][i].hexes[8] = self.hexes[j // 2][i - 1]
                        if i < h_length + d_length - int(abs((d_length * 4 - 1) / 2 - j + 1) + 1) // 2:
                            self.corners[j][i].hexes[4] = self.hexes[j // 2][i]

    def assign_adjacent(self):
        self.assign_edges_to_hexes()
        self.assign_corners_to_hexes()
        self.assign_hexes_to_edges()
        self.assign_hexes_to_corners()

    def __repr__(self):
        return '\n'.join([str({i: hexes}) for i, hexes in self.hexes.items()]) + '\n\n' + '\n'.join(
            [str({i: edges}) for i, edges in self.edges.items()]) + '\n\n' + '\n'.join(
            [str({i: corners}) for i, corners in self.corners.items()])


class CatanHexBoard(HexBoard):
    def __init__(self, max_players=4):
        self.resource_types = ['ore', 'brick', 'wool', 'grain', 'lumber', 'desert']
        if max_players == 4:
            super().__init__(3, 3)
            self.resource_count = dict(zip(self.resource_types, [3, 3, 4, 4, 4, 1]))
            self.numbers = reduce(list.__add__,
                                  map(lambda x: [x, x] if abs(x - 7) < 5 else [x], list(range(2, 7)) + list(range(8, 13))))
        elif max_players == 6:
            super().__init__(3, 4)
            self.resource_count = dict(zip(self.resource_types, [5, 5, 6, 6, 6, 2]))
            self.numbers = reduce(list.__add__,
                                  map(lambda x: [x, x, x] if abs(x - 7) < 5 else [x, x], list(range(2, 7)) + list(range(8, 13))))
        self.players = ['orange red', 'blue', 'floral white']
        self.resources = reduce(list.__add__, [[resource] * repeat for resource, repeat in self.resource_count.items()])
        self.assign_resources()
        self.assign_numbers()

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
            {y: CatanEdge((y, x)) for y in range(h_length * 2 + (d_length - 1 - abs(d_length * 2 - 1 - x) // 2) * 2)}),
                              range(d_length * 4 - 1)))
        self.corners = dict(map(lambda x: (x, {y: CatanCorner((y, x)) for y in range(
            h_length + d_length - int(abs((d_length * 4 - 1) / 2 - x) + 1) // 2)}), range(d_length * 4)))
