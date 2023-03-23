import random
from functools import reduce


def from_camel(phrase):
    '''
    convert Python's case convention (aa_bb_cc) into Class case convention (AaBbCc)
    with the exception of word AI
    '''
    def title_ai(word):
        if word == 'ai':
            return 'AI'
        else:
            return word.title()

    return ''.join(list(map(title_ai, phrase.split('_'))))


def pips_weight(corner, *args):
    '''
    weight for potential settlements and cities
    '''
    return corner.pips()


def generic_harbor_weight(corner, player_pips):
    '''
    weight for potential settlements and cities
    '''
    if corner.harbor == 'x':
        resource_pips = corner.resource_pips()
        total_pips = {resource: player_pips[resource] + resource_pips[resource] for resource in player_pips.keys()}
        max_pips = max([(resource, pips) for resource, pips in total_pips.items()], key=lambda x: x[1])[1]
        return max_pips
    else:
        return 0


def resource_harbor_weight(corner, player_pips):
    '''
    weight for potential settlements and cities
    '''
    if corner.harbor is not None and corner.harbor != 'x':
        resource_pips = corner.resource_pips()
        total_pips = {resource: player_pips[resource] + resource_pips[resource] for resource in player_pips.keys()}

        return total_pips[corner.harbor]
    else:
        return 0


def diversity_weight(corner, player_pips):
    '''
    weight for potential settlements and cities
    '''
    def diversity_score(pips):
        for key in pips.keys():
            pips[key] += 1
        average = reduce(int.__add__, [pip for pip in pips.values()]) / 5
        return 1 - (reduce(float.__mul__, [pip / average for pip in pips.values()]) ** (1 / 5))

    resource_pips = corner.resource_pips()
    total_pips = {resource: player_pips[resource] + resource_pips[resource] for resource in player_pips.keys()}
    score = - diversity_score(total_pips)
    return score


def random_weight(*args):
    '''
    weight for potentially anything
    '''
    return random.random()

def path_to_settlement(settlement, player, board):
    def not_blocked(corner, player):
        if corner.settlement is None:
            return corner.city is None or corner.city == player
        else:
            return corner.settlement == player
    player_roads = [edge for edge in board.edge_list if edge.road==player]
    player_corners = set([corner.coor for road in player_roads for corner in road.corners.values() if not_blocked(corner, player)])
    player_corners = [board.corners[coor[1]][coor[0]] for coor in player_corners]
    paths = [corner.road_path(settlement, player) for corner in player_corners]
    paths = min([path for path in paths if path[0] < 1000] + [(1000, [])], key=lambda x: x[0])
    return paths

