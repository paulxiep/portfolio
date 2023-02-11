import json
import operator
from functools import reduce
import os
import re
import shutil
from util import *
from get_save_data import get_save_data
from get_node_data import get_node_data

N_END_NODES = 3
all_mod_folder = '/Users/paulx/Documents/Paradox Interactive/Europa Universalis IV/mod'
mod_path = '/Users/paulx/Documents/Paradox Interactive/Europa Universalis IV/mod/dynamic_trade'
node_data_path = 'common/tradenodes'

def rank_nodes(countries, node_data):
    all_nodes = {}
    order = 1
    for value_pair in countries.values():
        if value_pair[1] is not None:
            try:
                country_node = list(filter(lambda x: value_pair[1] in x[1],
                                    [(key, value['members']) for key, value in node_data.items()]))[0][0]
                if country_node not in all_nodes.keys():
                    all_nodes[country_node] = order
                    order += 1
            except:
                print(value_pair)
    return all_nodes


def make_outgoing(node_data, node_rank, end_nodes):
    def min_with_index(iterator):
        min_index, min_value = min(enumerate(iterator), key=operator.itemgetter(1))
        return min_index * 10 + (20 - min_value) * 1000
    distance = 1
    nodes_at_distance = [[end_node] for end_node in end_nodes]
    for i, end_node in enumerate(end_nodes):
        node_data[end_node]['distance'] = [0 if j==i else 1 for j in range(N_END_NODES)]
        node_data[end_node]['outgoing'] = []
    all_nodes = [[] for _ in range(N_END_NODES)]
    while any([len(nodes_at_distance[i]) > 0 for i in range(N_END_NODES)]):
        for i in range(N_END_NODES):
            print(distance, nodes_at_distance[i])
            new_nodes_at_distance = []
            for node in nodes_at_distance[i]:
                for connection in node_data[node]['node_connections']:
                    if node_data[connection[0]].get('distance', None) is None:
                        node_data[connection[0]]['distance'] = [None, None, None]
                    if isinstance(node_data[connection[0]]['distance'], list) and node_data[connection[0]]['distance'][
                        i] is None:
                        node_data[connection[0]]['distance'][i] = distance
                    if connection[0] not in all_nodes[i]:
                        new_nodes_at_distance.append(connection[0])
                        all_nodes[i].append(connection[0])
                    if ((node_data[connection[0]]['distance'][i] > node_data[node]['distance'][i] \
                         and not any([node_data[connection[0]]['distance'][j] is not None \
                                      and node_data[connection[0]]['distance'][j] <
                                      node_data[connection[0]]['distance'][i] for j in range(0, i)])) \
                        or \
                        (node_data[connection[0]]['distance'][i] == node_data[node]['distance'][i] and
                         float(node_data[connection[0]]['trade_power']) < float(node_data[node]['trade_power']))\
                        and not any([node_data[connection[0]]['distance'][j] is not None \
                                     and node_data[connection[0]]['distance'][j] < node_data[connection[0]]['distance'][
                                         i] for j in range(0, i)])) \
                            and node_data[node]['node_connections'][
                        node_data[node]['node_connections'].index(connection[0])] not in node_data[node].get('outgoing',
                                                                                                             []):

                        node_data[connection[0]]['outgoing'] = node_data[connection[0]].get('outgoing', []) + [
                            node_data[connection[0]]['node_connections'][
                                node_data[connection[0]]['node_connections'].index(node)]]
                        print(f'{connection[0]}->{node}')
            nodes_at_distance[i] = new_nodes_at_distance.copy()
        distance += 1
    for v in node_data.values():
        v.pop('node_connections')
        v.pop('outgoing_nodes')
        v.pop('outgoing_paths')
        v.pop('outgoing_control')
        v.pop('incoming_nodes')
        v.pop('incoming_paths')
        v.pop('incoming_control')
    return dict(sorted(node_data.items(), key=lambda x: min_with_index([(x[1]['distance'][i]) for i in range(N_END_NODES)]) \
                                + x[1]['trade_power']))

def gen_nodes_text(node_data):
    def gen_node_text(k, v):
        out = f'{k}=' + '{\n\tlocation=' + v['location'][0] + '\n\tinland=yes' * v['inland'] + '\n\t' \
              + reduce(str.__add__, ['outgoing={\n\t\tname=' + outgoing[0] \
                                     + '\n\t\tpath={\n\t\t\t' + ' '.join(outgoing[1]) \
                                     + '\n\t\t}' \
                                     + '\n\t\tcontrol={\n\t\t\t' + ' '.join(outgoing[2]) \
                                     + '\n\t\t}' \
                                     + '\n\t}\n\t' for outgoing in set(v['outgoing'])], '') \
              + 'members={\n\t\t' + ' '.join(v['members']) + '\n\t}\n' + '\tend=yes\n' * int(
            len(v['outgoing']) == 0) + '}\n'

        return out

    out = ''
    for k, v in node_data.items():
        out += gen_node_text(k, v)
    print('nodes_text generated')
    return out

if __name__ == '__main__':
    node_data, countries = get_node_data()
    with open('node_data.json', 'w') as f:
        json.dump(node_data, f)
    with open('countries.json', 'w') as f:
        json.dump(countries, f)
    node_rank = rank_nodes(countries, node_data)
    print('top 10 nodes:', list(node_rank.keys())[:10])
    end_nodes = list(node_rank.keys())[:N_END_NODES]
    node_data = make_outgoing(node_data, node_rank, end_nodes)
    nodes_text = gen_nodes_text(node_data)
    if not os.path.exists(os.path.join(mod_path, node_data_path)):
        os.makedirs(os.path.join(mod_path, node_data_path))
    with open(os.path.join(mod_path, node_data_path, '00_tradenodes.txt'), 'w') as f:
        f.write(nodes_text)
    shutil.copy('descriptor.mod', os.path.join(mod_path, 'descriptor.mod'))
    shutil.copy('dynamic_trade.mod', os.path.join(all_mod_folder, 'dynamic_trade.mod'))