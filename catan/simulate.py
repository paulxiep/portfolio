from catan_simulation import CatanSimulation
from multiprocessing import Pool
import os
from catan_units.ai_choices import sub_ais

def simulate(x):
    personality = {sub_ai: 'basis' for sub_ai in sub_ais}
    personality['initial_road_ai'] = 'random'
    # personality['first_settlement_ai'] = 'basis'
    # personality['second_settlement_ai'] = 'basis'
    personality['road_ai'] = 'random'
    player = ['white', 'blue', 'orange']
    color = ['white', 'blue', 'orange']
    ai = [True, True, True]
    personalities = [personality, personality, personality]
    CatanSimulation(list(zip(player, color, ai, personalities)),
                    actionlogfile=f'action_logs/action_{x}.txt',
                    statelogfile=f'state_logs/state_{x}.txt',
                    optionlogfile=f'option_logs/option_{x}.txt').run_simulation()

if __name__ == '__main__':
    simulations = range(1, 12001)
    with Pool(os.cpu_count()*3//4) as p:
        p.map(simulate, simulations)
    # simulate(1)