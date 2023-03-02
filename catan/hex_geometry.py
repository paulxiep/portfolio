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
    def __init__(self, coor=None, d_length=3):
        super().__init__(coor)
        if coor[1] % 2 == 1:
            self.type = '|'
        else:
            if coor[1]<2*d_length-1:
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