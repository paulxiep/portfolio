from catan_simulation import CatanSimulation
from multiprocessing import Pool
import os
from catan_units.ai_choices import sub_ais

to_train = 'road'
def simulate(x):
    personality = {sub_ai: 'basis' for sub_ai in sub_ais}
    personality['road_ai'] = 'primitive_ml_basis'
    personality['road_ai'] = 'random'
    player = ['blue', 'red', 'white', 'orange']
    color = ['blue', 'red', 'white', 'orange']
    ai = [True, True, True, True]
    personalities = [personality, personality, personality, personality]
    CatanSimulation(list(zip(player, color, ai, personalities)),
                    actionlogfile=f'ai_logs/{to_train}_ai_logs/action_logs/action_{x}.txt',
                    statelogfile=f'ai_logs/{to_train}_ai_logs/state_logs/state_{x}.txt',
                    optionlogfile=f'ai_logs/{to_train}_ai_logs/option_logs/option_{x}.txt').run_simulation()

if __name__ == '__main__':
    simulations = range(96001, 120001)
    with Pool(os.cpu_count()*3//4) as p:
        p.map(simulate, simulations)
    # simulate(1)