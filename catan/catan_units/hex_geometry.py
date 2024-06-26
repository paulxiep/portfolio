from functools import reduce

'''
The __repr__ changes all the time based on my debugging needs at each point
'''


class Geometry:
    def __init__(self, coor=None):
        self.coor = coor

    def __repr__(self):
        return str(self.coor)

    def shift_coor(self):
        '''
        this function is for converting Python indexing (starts at 0) into more human-readable indexing.
        '''
        self.coor = (self.coor[0] + 1, self.coor[1] + 1)
        return self


class Hex(Geometry):
    def __init__(self, coor=None):
        '''
        define potential edge and corner connections with the keys as clock direction
        '''
        super().__init__(coor)
        self.edges = {i * 2 - 1: None for i in range(1, 7)}
        self.corners = {i * 2: None for i in range(1, 7)}


class Edge(Geometry):
    def __init__(self, coor=None, d_length=3):
        super().__init__(coor)
        if coor[1] % 2 == 1:
            self.type = '|'
        else:
            if coor[1] < 2 * d_length - 1:
                if coor[0] % 2 == 0:
                    self.type = '/'
                else:
                    self.type = '\\'
            else:
                if coor[0] % 2 == 1:
                    self.type = '/'
                else:
                    self.type = '\\'
        self.hexes = {'left': None, 'right': None}
        self.corners = {'top': None, 'bottom': None}
        self.edges = {'top_left': None, 'top_right': None, 'bottom_left': None, 'bottom_right': None}

    def __repr__(self):
        return self.type + str(self.coor)


class Corner(Geometry):
    def __init__(self, coor=None):
        super().__init__(coor)
        if coor[1] % 2 == 0:
            self.type = 'top'
            self.hexes = {4 * i - 2: None for i in range(1, 4)}
            self.edges = {4 * i: None for i in range(1, 4)}
        else:
            self.type = 'bottom'
            self.hexes = {4 * i: None for i in range(1, 4)}
            self.edges = {4 * i - 2: None for i in range(1, 4)}

    def approaching(self, interim, other):
        '''
        for used in self.distance method
        '''
        a, b = self.coor
        i, j = interim.coor
        x, y = other.coor

        # print(a, b, i, j, x, y)
        def horizontal_approach(b, j):
            if self.type == 'bottom':
                return b - j == 1
            else:
                return j - b == 1

        if y == b:
            out = (i - a) * (x - a) >= 0 and horizontal_approach(b, j)
        elif x == a:
            out = i == a and (j - b) * (y - b) >= 0
        else:
            if self.type == 'bottom':
                if b <= 2 * self.board.d_length:
                    out = (((i - a == 0 and x - a >= 0) or
                            (i - a < 0 and x - a < 0)) and (j - b) == -1) or \
                          ((i - a) == 0 and (j - b) * (y - b) > 0 and (j - b) > 0)
                else:
                    out = (((i - a == 0 and x - a <= 0) or
                            (i - a > 0 and x - a > 0)) and (j - b) == -1) or \
                          ((i - a) == 0 and (j - b) * (y - b) > 0 and (j - b) > 0)
            else:
                if b <= 2 * self.board.d_length:
                    out = (((i - a == 0 and x - a <= 0) or
                            (i - a > 0 and x - a > 0)) and (j - b) == 1) or \
                          ((i - a) == 0 and (j - b) * (y - b) > 0 and (j - b) < 0)
                else:
                    out = (((i - a == 0 and x - a >= 0) or
                            (i - a < 0 and x - a < 0)) and (j - b) == 1) or \
                          ((i - a) == 0 and (j - b) * (y - b) > 0 and (j - b) < 0)

        return out

    def distance(self, other, accu=[], ori=None):
        '''
        measures distance between self and other corner
        '''

        def new_ori(self, ori):
            if ori is not None:
                return ori
            else:
                return self.coor

        if self.coor == other.coor:
            return len(accu)
        else:
            dist = min(
                [corner.distance(other, accu + [self.coor], new_ori(self, ori)) for edge in self.edges.values() if
                 edge is not None for corner in edge.corners.values() if
                 corner is not None and corner.coor != self.coor and corner.coor not in accu and self.approaching(
                     corner, other)] + [1000])
            return dist


class HexBoard:
    '''
    A hex board is defined as hex tiles arranged in the shape of a bigger hex
    If a hex field (hex tiles arranged as a rectangle) is ever needed, it should a be separate class
    Also if edges and corners aren't needed, then a new class should be defined too

    the many 'assign' methods should be self-explanatory
    '''
    def __init__(self, h_length, d_length):
        '''
        :param h_length: number of tiles in the top row
        :param d_length: number of tiles on one side edge (assuming vertical symmetry)
        '''
        if not hasattr(self, 'hex_class'):
            self.hex_class = Hex
            self.edge_class = Edge
            self.corner_class = Corner
        self.h_length = h_length
        self.d_length = d_length
        self.generate_elements()
        self.shift_coors()
        # for j in self.edges.keys():
        #     if j % 2 == 0:
        #         self.edges[j] = {k: self.edges[j][k] for k in
        #                          list(self.hexes[j // 2].keys()) + [len(self.hexes[j // 2].keys()) + 1]}
        self.hex_list = reduce(list.__add__, [list(v.values()) for v in self.hexes.values()])
        self.edge_list = reduce(list.__add__, [list(v.values()) for v in self.edges.values()])
        self.corner_list = reduce(list.__add__, [list(v.values()) for v in self.corners.values()])
        for corner in self.corner_list:
            corner.board = self
        self.assign_adjacent()

    def row_edge_count(self, j):
        if j % 2 == 0:
            return range(self.h_length * 2 + (self.d_length - 1 - abs(self.d_length * 2 - 1 - j) // 2) * 2)
        else:
            return range(self.h_length + self.d_length - 1 - abs(self.d_length - j//2 - 1) + 1)

    def generate_elements(self):
        '''
        generates actual hexes, edges, and corners elements of the hex board
        '''

        h_length = self.h_length
        d_length = self.d_length
        self.hexes = dict(
            map(lambda x: (x, {y: self.hex_class((y, x)) for y in range(h_length + d_length - 1 - abs(d_length - x - 1))}),
                range(2 * d_length - 1)))

        self.edges = dict(map(lambda x: (
            x, {y: self.edge_class((y, x), self.d_length) for y in
                self.row_edge_count(x)}),
                              range(d_length * 4 - 1)))
        # if y % 2 == 1:
        #     self.edges[y] = {x: self.edges[y][x] for x in
        #                      list(self.hexes[y // 2].keys()) + [len(self.hexes[y // 2].keys())]}
        # self.edges = dict(map(lambda x: (
        #     x, {y: Edge((y, x), self.d_length) for y in
        #         range(h_length * 2 + (d_length - 1 - abs(d_length * 2 - 1 - x) // 2) * 2)}),
        #                       range(d_length * 4 - 1)))
        self.corners = dict(map(lambda x: (
            x, {y: self.corner_class((y, x)) for y in range(h_length + d_length - int(abs((d_length * 4 - 1) / 2 - x) + 1) // 2)}),
                                range(d_length * 4)))

    def shift_coors(self):
        '''
        used to convert Python's 0-indexing into human-friendly 1-indexing
        '''
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
                self.hexes[j][i].edges[3] = self.edges[j * 2][i + 1]
                self.hexes[j][i].edges[9] = self.edges[j * 2][i]
                if j < d_length:
                    self.hexes[j][i].edges[11] = self.edges[j * 2 - 1][2 * i - 1]
                    self.hexes[j][i].edges[1] = self.edges[j * 2 - 1][2 * i]
                    self.hexes[j][i].edges[5] = self.edges[j * 2 + 1][2 * i + 1]
                    self.hexes[j][i].edges[7] = self.edges[j * 2 + 1][2 * i]
                elif j == d_length:
                    self.hexes[j][i].edges[11] = self.edges[j * 2 - 1][2 * i - 1]
                    self.hexes[j][i].edges[1] = self.edges[j * 2 - 1][2 * i]
                    self.hexes[j][i].edges[5] = self.edges[j * 2 + 1][2 * i]
                    self.hexes[j][i].edges[7] = self.edges[j * 2 + 1][2 * i - 1]
                else:
                    self.hexes[j][i].edges[11] = self.edges[j * 2 - 1][2 * i]
                    self.hexes[j][i].edges[1] = self.edges[j * 2 - 1][2 * i + 1]
                    self.hexes[j][i].edges[5] = self.edges[j * 2 + 1][2 * i]
                    self.hexes[j][i].edges[7] = self.edges[j * 2 + 1][2 * i - 1]

    def assign_corners_to_hexes(self):
        h_length = self.h_length
        d_length = self.d_length
        for j in range(1, 2 * d_length):
            for i in range(1, h_length + d_length - abs(d_length - j - 1 + 1)):
                if j < d_length:
                    self.hexes[j][i].corners[12] = self.corners[j * 2 - 1][i]
                    self.hexes[j][i].corners[10] = self.corners[j * 2][i]
                    self.hexes[j][i].corners[2] = self.corners[j * 2][i + 1]
                    self.hexes[j][i].corners[8] = self.corners[j * 2 + 1][i]
                    self.hexes[j][i].corners[4] = self.corners[j * 2 + 1][i + 1]
                    self.hexes[j][i].corners[6] = self.corners[j * 2 + 2][i + 1]
                elif j == d_length:
                    self.hexes[j][i].corners[12] = self.corners[j * 2 - 1][i]
                    self.hexes[j][i].corners[10] = self.corners[j * 2][i]
                    self.hexes[j][i].corners[2] = self.corners[j * 2][i + 1]
                    self.hexes[j][i].corners[8] = self.corners[j * 2 + 1][i]
                    self.hexes[j][i].corners[4] = self.corners[j * 2 + 1][i + 1]
                    self.hexes[j][i].corners[6] = self.corners[j * 2 + 2][i]
                else:
                    self.hexes[j][i].corners[12] = self.corners[j * 2 - 1][i + 1]
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
                                self.edges[j][i].hexes['right'] = self.hexes[j // 2][(i + 1) // 2]
                        if j < d_length * 4 - 1:
                            if i > 1:
                                self.edges[j][i].hexes['left'] = self.hexes[(j + 1) // 2][(i) // 2]
            else:
                for i in range(1, h_length + d_length - abs(d_length - j // 2 - 1 + 1) + 1):
                    # print('here', i, j)
                    if i > 1:
                        self.edges[j][i].hexes['left'] = self.hexes[j // 2][i - 1]
                    if i < h_length + d_length - abs(d_length - j // 2 - 1 + 1):
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
                        # if i :
                        if j <= d_length * 2:
                            if 1 < i < h_length + d_length - int(abs((d_length * 4 - 1) / 2 - j + 1) + 1) // 2:
                                self.corners[j][i].hexes[12] = self.hexes[(j - 2) // 2][i - 1]
                        elif i < h_length + d_length + 1 - int(abs((d_length * 4 - 1) / 2 - j + 1) + 1) // 2:
                            self.corners[j][i].hexes[12] = self.hexes[(j - 2) // 2][i]
                    if j < d_length * 4:
                        if i > 1:
                            self.corners[j][i].hexes[8] = self.hexes[j // 2][i - 1]
                        if i < h_length + d_length - int(abs((d_length * 4 - 1) / 2 - j + 1) + 1) // 2:
                            self.corners[j][i].hexes[4] = self.hexes[j // 2][i]

    def assign_edges_to_corners(self):
        for corner in self.corner_list:
            if corner.type == 'top':
                try:
                    corner.edges[4] = corner.hexes[6].edges[1]
                    corner.edges[8] = corner.hexes[6].edges[11]
                except:
                    try:
                        corner.edges[8] = corner.hexes[10].edges[5]
                    except:
                        pass
                    try:
                        corner.edges[4] = corner.hexes[2].edges[7]
                    except:
                        pass
                try:
                    try:
                        corner.edges[12] = corner.hexes[2].edges[9]
                    except:
                        corner.edges[12] = corner.hexes[10].edges[3]
                except:
                    pass
            else:
                try:
                    corner.edges[2] = corner.hexes[12].edges[5]
                    corner.edges[10] = corner.hexes[12].edges[7]
                except:
                    try:
                        # try:
                        corner.edges[2] = corner.hexes[4].edges[11]
                    except:
                        pass
                    try:
                        corner.edges[10] = corner.hexes[8].edges[1]
                    except:
                        pass
                try:
                    try:
                        corner.edges[6] = corner.hexes[4].edges[9]
                    except:
                        corner.edges[6] = corner.hexes[8].edges[3]
                except:
                    pass

    def assign_corners_to_edges(self):
        for edge in self.edge_list:
            if edge.type == '/':
                try:
                    edge.corners['top'] = edge.hexes['left'].corners[4]
                    edge.corners['bottom'] = edge.hexes['left'].corners[6]
                except:
                    edge.corners['top'] = edge.hexes['right'].corners[12]
                    edge.corners['bottom'] = edge.hexes['right'].corners[10]
            elif edge.type == '\\':
                try:
                    edge.corners['top'] = edge.hexes['left'].corners[12]
                    edge.corners['bottom'] = edge.hexes['left'].corners[2]
                except:
                    edge.corners['top'] = edge.hexes['right'].corners[8]
                    edge.corners['bottom'] = edge.hexes['right'].corners[6]
            else:
                try:
                    edge.corners['top'] = edge.hexes['left'].corners[2]
                    edge.corners['bottom'] = edge.hexes['left'].corners[4]
                except:
                    edge.corners['top'] = edge.hexes['right'].corners[10]
                    edge.corners['bottom'] = edge.hexes['right'].corners[8]

    def assign_edges_to_edges(self):
        for edge in self.edge_list:
            if edge.type == '/':
                edge.edges['top_left'] = edge.corners['top'].edges[12] if edge.corners['top'].edges[
                                                                              12] is not None else None
                edge.edges['top_right'] = edge.corners['top'].edges[4] if edge.corners['top'].edges[
                                                                              4] is not None else None
                edge.edges['bottom_right'] = edge.corners['bottom'].edges[6] if edge.corners['bottom'].edges[
                                                                                    6] is not None else None
                edge.edges['bottom_left'] = edge.corners['bottom'].edges[10] if edge.corners['bottom'].edges[
                                                                                    10] is not None else None
            if edge.type == '\\':
                edge.edges['top_left'] = edge.corners['top'].edges[8] if edge.corners['top'].edges[
                                                                             8] is not None else None
                edge.edges['top_right'] = edge.corners['top'].edges[12] if edge.corners['top'].edges[
                                                                               12] is not None else None
                edge.edges['bottom_right'] = edge.corners['bottom'].edges[2] if edge.corners['bottom'].edges[
                                                                                    2] is not None else None
                edge.edges['bottom_left'] = edge.corners['bottom'].edges[6] if edge.corners['bottom'].edges[
                                                                                   6] is not None else None
            if edge.type == '|':
                edge.edges['top_left'] = edge.corners['top'].edges[10] if edge.corners['top'].edges[
                                                                              10] is not None else None
                edge.edges['top_right'] = edge.corners['top'].edges[2] if edge.corners['top'].edges[
                                                                              2] is not None else None
                edge.edges['bottom_right'] = edge.corners['bottom'].edges[4] if edge.corners['bottom'].edges[
                                                                                    4] is not None else None
                edge.edges['bottom_left'] = edge.corners['bottom'].edges[8] if edge.corners['bottom'].edges[
                                                                                   8] is not None else None

    def assign_adjacent(self):
        self.assign_edges_to_hexes()
        self.assign_corners_to_hexes()
        self.assign_hexes_to_edges()
        self.assign_hexes_to_corners()
        self.assign_edges_to_corners()
        self.assign_corners_to_edges()
        self.assign_edges_to_edges()

    def __repr__(self):
        return '\n'.join([str({i: hexes}) for i, hexes in self.hexes.items()]) + '\n\n' + '\n'.join(
            [str({i: edges}) for i, edges in self.edges.items()]) + '\n\n' + '\n'.join(
            [str({i: corners}) for i, corners in self.corners.items()])
