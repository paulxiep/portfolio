import os
import json
import cv2
import re
import numpy as np
from functools import reduce
from collections import defaultdict

data_dir = '/Games/Steam/steamapps/common/Victoria 3/game'
map_data_file = os.path.join(data_dir, 'map_data/provinces.png')
resource_data_dir = os.path.join(data_dir, 'map_data/state_regions')
pop_data_dir = os.path.join(data_dir, 'common/history/pops')
state_data_file = os.path.join(data_dir, 'common/history/states/00_states.txt')
terrain_file = os.path.join(data_dir, 'map_data/province_terrains.txt')
building_dir = os.path.join(data_dir, 'common/history/buildings')

def get_map():
    def to_province_id(x):
        return 'x' + hex(x[2])[2:].upper().zfill(2) + hex(x[1])[2:].upper().zfill(2) + hex(x[0])[2:].upper().zfill(2)
    game_map = cv2.imread(map_data_file)
    return np.apply_along_axis(to_province_id, 2, game_map)

def get_provinces_terrain(redo=False, drop_ocean=False):
    if not redo and os.path.exists('vic3_provinces.json'):
        with open('vic3_provinces.json', 'r') as f:
            return json.load(f)

    else:
        with open(terrain_file, 'r') as f:
            provinces = f.readlines()[1:]
        provinces = list(map(lambda x: x[:-2].split('="'), provinces))

        province_map = get_map() if 'province_map' not in globals() else province_map
        connections = defaultdict(set)
        for i in range(province_map.shape[0]):
            for j in range(province_map.shape[1]):
                if i < province_map.shape[0]-1 and province_map[i, j] != province_map[i+1 ,j]:
                    connections[province_map[i, j]].add(province_map[i+1, j])
                    connections[province_map[i+1, j]].add(province_map[i, j])
                if j < province_map.shape[1]-1 and province_map[i, j] != province_map[i, j+1]:
                    connections[province_map[i, j]].add(province_map[i, j+1])
        oceans = tuple(zip(*list(filter(lambda x: x[1 ]=='ocean', provinces))))[0]
        coast = set()
        for ocean in oceans:
            for item in connections[ocean]:
                if item not in oceans:
                    coast.add(item)
        print(f'found {len(coast)} coast out of {len(provinces)} provinces.')

        def province_json(prov):
            if prov[1] != 'ocean':
                return {prov[0]: {'terrain': prov[1], 'coast': prov[0] in coast}}
            elif not drop_ocean:
                return {prov[0]: {'terrain': prov[1]}}
            else:
                return {}
        provinces = reduce(lambda x, y: {**x, **y}, map(province_json, provinces), {})
        with open('vic3_provinces.json', 'w') as f:
            json.dump(provinces, f)
        return provinces

def get_states(redo=False):
    def process_state(state):
        def process_sub(sub):
            sub = sub.split('country=c:')[1]
            return \
                {sub[:3]: list(map(lambda x: 'x' + x, sub.split('{')[1].split('}')[0].replace('"', '').split('x')[1:]))}

        state = state.replace('\n', '').replace('\t', '').replace(' ', '')
        state_name, state_sub = tuple(state.split('=', 1))
        state_sub = state_sub.split('create_state=')[1:]
        state_sub = map(process_sub, state_sub)
        return {state_name: reduce(lambda x, y: {**x, **y}, state_sub, {})}

    if not redo and os.path.exists('vic3_states.json'):
        with open('vic3_states.json', 'r') as f:
            return json.load(f)
    else:
        with open(os.path.join(data_dir, 'common', 'history', 'states', '00_states.txt'), 'r') as f:
            states = ('\n'.join(f.read().split('\n\n')[1:])[12:-3]).split('s:')[1:]
        states = reduce(lambda x, y: {**x, **y}, map(process_state, states), {})
        with open('vic3_states.json', 'w') as f:
            json.dump(states, f)
        return states


def remove_comment(line):
    return line.split('#')[0]


def get_state_pops(redo=False):
    def process_file(region):
        def process_pop(data):
            data = '\n'.join(list(map(remove_comment, data.split('\n'))))
            states = data[15:-2].replace('\n', '').replace('\t', '').replace(' ', '').replace('pop_type=slaves',
                                                                                              '').split('s:')

            def process_state(state):
                #         print('--------\n', state)
                state_name, state = tuple(state.split('=', 1))
                regions = state.split('region_state:')[1:]

                def process_region(region):
                    region_name = region[:3]
                    region = region[3:].split('size=')[1:]

                    def process_pop(pop):
                        return int(pop.split('}')[0])

                    return {region_name: reduce(int.__add__, map(process_pop, region))}

                regions = reduce(lambda x, y: {**x, **y}, map(process_region, regions))
                return {state_name: regions}

            data = reduce(lambda x, y: {**x, **y}, map(process_state, filter(lambda x: len(x) > 0, states)), {})
            return data

        with open(os.path.join(pop_data_dir, region), 'r') as f:
            region_state_pops = process_pop(f.read())
        # states = get_states()
        for key in region_state_pops:
            region_state_pops[key] = {'region': region,
                                      'pop': reduce(int.__add__, [value for value in region_state_pops[key].values()])}
        return region_state_pops

    state_pops = reduce(lambda x, y: {**x, **y},
                        map(process_file, filter(lambda x: x.find('sea') < 0, os.listdir(pop_data_dir))))
    json.dump(state_pops, open('vic3_state_pops.json', 'w'))
    return state_pops


def get_state_resources(redo=False):
    traits = ['malaria', 'severe_malaria', 'soil', 'terra', 'harbor',
              'whaling', 'fishing', 'river', 'delta', 'desert', 'mountain', 'valley', 'forest', 'lake']
    traits = {trait: False for trait in traits}

    def process_file(region):
        def process_resource(data):
            data = '\n'.join(list(map(remove_comment, data.split('\n'))))
            states = data[3:].split('STATE_')[1:]

            def process_state(state):
                state_name, state_data = tuple(state.split(' = ', 1))
                state_name = 'STATE_' + state_name
                state_data = state_data.split('subsistence_building = "', 1)[1]
                subsistence_building, state_data = tuple(state_data.split('"', 1))
                state_data = state_data.split('impassable = { ')
                if len(state_data) > 1:
                    state_data = state_data[1]
                    impassables, state_data = tuple(state_data.split(' }', 1))
                    impassables = impassables.replace('"', '').split(' ')
                    state_data = state_data.split('traits = {')[-1]
                else:
                    impassables = []
                    state_data = state_data[0].split('traits = {')[-1]
                state_traits, state_data = tuple(state_data.split('arable_land = '))
                this_state_traits = traits.copy()
                for trait in traits.keys():
                    this_state_traits[trait] = trait in state_traits
                if this_state_traits['severe_malaria']:
                    this_state_traits['malaria'] = False
                if this_state_traits['terra']:
                    this_state_traits['soil'] = True
                this_state_traits.pop('terra')
                arable_land, state_data = tuple(state_data.split('\n', 1))
                arable_land = int(arable_land)
                state_data = state_data.split('arable_resources = ')[1]
                arable_resources, state_data = tuple(state_data.split('\n', 1))
                arable_resources = json.loads(
                    arable_resources.replace('" "', '", "').replace('{', '[').replace('}', ']'))
                try:
                    state_data = state_data.split('capped_resources = {')[1].split('}')[0] + '}'
                    capped_resources = json.loads(
                        '{' + state_data.replace(' ', '').replace('\n', '').replace('b', ',"b').replace('=', '":')[1:])
                except:
                    capped_resources = {}
                return {state_name: {'subsistence_building': subsistence_building,
                                     'state_traits': this_state_traits,
                                     'arable_land': arable_land,
                                     'arable_resources': arable_resources,
                                     'capped_resources': capped_resources,
                                     'impassables': impassables}}

            return reduce(lambda x, y: {**x, **y}, map(process_state, states))

        region_state_resource = process_resource(open(os.path.join(resource_data_dir, region), 'r').read())
        for key in region_state_resource:
            region_state_resource[key] = {'region': region, **region_state_resource[key]}
        return region_state_resource

    state_resources = reduce(lambda x, y: {**x, **y},
                             map(process_file, filter(lambda x: x.find('sea') < 0, os.listdir(resource_data_dir))))
    json.dump(state_resources, open('vic3_state_resources.json', 'w'))
    return state_resources


def get_state_buildings(redo=False):
    def process_file(region):
        def process_buildings(data):
            data = '\n'.join(list(map(remove_comment, data.split('\n'))))
            states = data[3:].replace(' ', '').split('s:')[1:]

            def process_state(state):
                state_name, buildings = tuple(state.split('=', 1))
                #         state_name = 'STATE_' + state_name
                try:
                    buildings = buildings.split('building_')[1:]

                    def process_building(building):
                        building = building.split('"')[0]
                        return building

                    def add_defaultdict(x, y):
                        x[y] = x[y] + 1
                        return x

                    all_buildings = defaultdict(int)
                    return {state_name: reduce(add_defaultdict, map(process_building, buildings), all_buildings)}
                except:
                    return {state_name: defaultdict(int)}

            return reduce(lambda x, y: {**x, **y}, map(process_state, states))

        with open(os.path.join(building_dir, region), 'r') as f:
            region_state_buildings = process_buildings(f.read())
        region_state_buildings['STATE_JETISY'] = {}
        region_state_buildings['STATE_SHENGJING'] = {}
        # states = get_states()
        for key in region_state_buildings:
            region_state_buildings[key] = {'region': region, 'buildings': region_state_buildings[key]}
        return region_state_buildings

    state_buildings = reduce(lambda x, y: {**x, **y},
                             map(process_file, filter(lambda x: x.find('sea') < 0, os.listdir(building_dir))))
    with open('vic3_state_buildings.json', 'w') as f:
        json.dump(state_buildings, f)
    return state_buildings


def get_state_terrain(redo=False, drop_impassables=False):
    terrains = ['desert',
                'forest',
                'hills',
                'jungle',
                'lakes',
                'mountain',
                'plains',
                'savanna',
                'snow',
                'tundra',
                'wetland']

    def provinces_to_terrain(prov):
        out = {terrain: 0 for terrain in terrains}
        out['coast'] = 0
        for province in prov:
            province = (province[0] + province[1:].upper()).split('#')[0]
            if provinces[province]['terrain'] == 'ocean':
                continue
            out[provinces[province]['terrain']] += 1
            if provinces[province]['coast']:
                out['coast'] += 1
        return {'terrain': out}

    if not redo and os.path.exists('vic3_state_terrain.json'):
        with open('vic3_state_terrain.json', 'r') as f:
            return json.load(f)
    else:
        provinces = get_provinces_terrain()
        states = get_states()
        if drop_impassables:
            impassables = {k: v['impassables'] for k, v in get_state_resources().items()}
            # print(impassables)
            state_terrain = {k: provinces_to_terrain(
                list(filter(lambda x: x not in impassables[k], reduce(list.__add__, [value for value in v.values()])))) for
                             k, v in states.items()}
        else:
            state_terrain =  {k: provinces_to_terrain(
                reduce(list.__add__, [value for value in v.values()]))
                for
                k, v in states.items()}
        with open('vic3_state_terrain.json', 'w') as f:
            json.dump(state_terrain, f)
        return state_terrain

if __name__ == '__main__':
    get_state_terrain()