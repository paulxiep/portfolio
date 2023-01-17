import pandas as pd
from functools import reduce
import cv2
import seaborn as sns
import numpy as np

colors = {'Bus': 'deepskyblue', 'TrolleyBus': 'orange', 'Tram': 'rebeccapurple', 'Metro': 'limegreen',
              'Taxi': 'mediumseagreen',
              'Monorail': 'crimson', 'Train': 'orangered', 'Boat': 'gold', 'Cable Car': 'blue', 'Air': 'mediumorchid',
              'Short': 'steelblue', 'Long': 'chocolate', 'Niche': 'lightseagreen'}

def map_city(city):
    line_0 = list(map(lambda x: x.replace('\'', ''), city[0].split('#')[0].split('\',\'')))
    categories = line_0[1].split(',')
    if len(city)==2:
        line_0 = list(map(lambda x: int(x[0]) + int(x[1]), zip(line_0[0].split(','), city[1].replace('\'', '').split(','))))
        line_0[-1] //= 2
        line_0[-2] //= 2
    else:
        line_0 = list(map(int, line_0.split(',')))
    return line_0, categories

def map_categories(categories):
    return [int(x in categories) for x in ['archipelago', 'layered', 'singlebodied', 'flat', 'bumpy', 'valley', 'canal', 'international', 'european']]

def categorize(transport):
    if transport in ['Bus', 'TrolleyBus', 'Tram', 'Boat']:
        return 'Short'
    if transport in ['Metro', 'Monorail', 'Train']:
        return 'Long'
    if transport in ['Cable Car', 'Air', 'Taxi']:
        return 'Niche'

def process_data(data):
    cities = list(map(lambda x: x.split('\n'), data.split('\n\n')))
    data, categories = zip(*list(map(map_city, cities)))
    categories = list(map(map_categories, categories))

    data = pd.merge(pd.DataFrame(np.array(data),
                                 columns=['Bus', 'TrolleyBus', 'Tram', 'Metro', 'Train', 'Boat', 'Air', 'Monorail',
                                          'Cable Car', 'Taxi', 'Total', 'Population', 'Tiles']).reset_index(),
                    pd.DataFrame(np.array(categories),
                                 columns=['archipelago', 'layered', 'singlebodied', 'flat', 'bumpy', 'valley', 'canal',
                                          'international', 'european']).reset_index(),
                    left_on='index', right_on='index')
    data = data.drop('index', axis=1).drop('Population', axis=1).drop('Tiles', axis=1)

    return data

def plot(cities):
    sets = {'All': cities,
            'Archipelago': cities[cities['archipelago'] == 1],
            'Layered': cities[cities['layered'] == 1],
            'Single-bodied': cities[cities['singlebodied'] == 1],
            'Flat': cities[cities['flat'] == 1],
            'Bumpy': cities[cities['bumpy'] == 1],
            'Valley': cities[cities['valley'] == 1],
            'Canal': cities[cities['canal'] == 1],
            'International': cities[cities['international'] == 1],
            'European': cities[cities['european'] == 1]
            }

    for plot_type in ['individual', 'categories']:
        for key in sets.keys():
            selected = sets[key].copy()[['Bus', 'TrolleyBus', 'Tram', 'Metro', 'Train',
                                         'Boat', 'Air', 'Monorail', 'Cable Car', 'Taxi', 'Total']]

            cities_percent = selected.copy()
            for transport in cities_percent.columns:
                if transport != 'Total':
                    cities_percent[transport] = cities_percent[transport] / cities_percent['Total'] * 100
            cities_percent['Total'] = cities_percent['Total'] / cities_percent['Total'] * 100
            cities_square = cities_percent.copy()
            cities_square.drop('Total', axis=1, inplace=True)
            for transport in cities_square.columns:
                cities_square[transport] = cities_square[transport].map(lambda x: x ** 2)
            cities_square['Total'] = reduce(pd.Series.add,
                                            [cities_square[transport] for transport in cities_square.columns])
            for transport in cities_square.columns:
                cities_square[transport] = cities_square[transport] / cities_square[
                    'Total'] * 100  # .map(lambda x: "%.3g" % x)

            ncities = len(cities_square.index)

            out = pd.concat([selected.sum(), cities_percent.mean()], axis=1)
            out.columns = ['total_riders', 'mean_rider_percentage']
            total = out['total_riders']['Total']
            out = out.drop('Total')
            out['Category'] = out.index
            out['Category'] = out['Category'].map(categorize)
            out['Transport'] = out.index

            for col in ['total_riders', 'mean_rider_percentage']:
                out = out.sort_values(col, ascending=False)
                import matplotlib.pyplot as plt

                sns.set(font_scale=1.5)
                sns.set_style("whitegrid")
                bar, ax = plt.subplots(figsize=(10, 6))
                if plot_type == 'individual':
                    ax = sns.barplot(x=col, y='Transport', data=out,
                                     ci=None, palette=colors, orient='h')
                elif plot_type == 'categories':
                    pd.pivot_table(out, index='Category', columns='Transport', values=col).loc[
                        ['Niche', 'Long', 'Short']].plot.barh(stacked=True, ax=ax, legend=False, color=colors)
                if col != 'Total Riders':
                    ax.set_title(f"{key} Cities (riders average {int(total / ncities)}, {ncities} cities)", fontsize=20)
                    ax.set_xlabel(col)
                    ax.set_ylabel("Transport")
                    if plot_type == 'individual':
                        for rect in ax.patches:
                            ax.text(rect.get_width(), rect.get_y() + rect.get_height() / 2, "%.1f%%" % rect.get_width())
                else:
                    ax.set_title(f"{key} Cities (total riders {int(total)}, {ncities} cities)", fontsize=20)
                    ax.set_xlabel(col)
                    ax.set_ylabel("Transport")
                    if plot_type == 'individual':
                        for rect in ax.patches:
                            ax.text(rect.get_width(), rect.get_y() + rect.get_height() / 2, int(rect.get_width()))
                bar.savefig(f"files/results/{plot_type}_{col}_{key.replace('-', '').lower()}.jpg")

def combine_plots():
    for plot_type in ['individual', 'categories']:
        for col in ['total_riders', 'mean_rider_percentage']:
            images = [['archipelago', 'canal', 'valley'], ['singlebodied', 'layered', 'international'],
                      ['flat', 'bumpy', 'european']]
            images = list(map(lambda y: list(map(lambda x: cv2.imread(f'files/results/{plot_type}_{col}_{x}.jpg'), y)), images))

            cv2.imwrite(f'files/results/{plot_type}_{col}.jpg',
                        np.concatenate(list(map(lambda x: np.concatenate(x, axis=1), images)), axis=0))

if __name__ == '__main__':
    with open('files/data.txt', 'r') as f:
        data = f.read()

    cities = process_data(data)
    plot(cities)
    combine_plots()

