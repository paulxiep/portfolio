import numpy as np

def squash_idle(player_memory):
    '''
    used for squashing idle turns in memory
    '''
    choose_memory = []
    for i, memory in enumerate(player_memory):
        if memory[-1] == 0 and not isinstance(memory[0], str):
            partial = [memory[0]], list(memory[7:]), memory[1]
        elif memory[-1] == 1 and not isinstance(memory[1], str):
            action = [partial[2] + 80 * memory[1]]
        elif memory[-1] == 3:
            choose_memory.append(partial[0] + action + [memory[2]] + list(memory[3:7]) + partial[1])

    for j in range(3, 7):
        choose_memory[-1][j] = player_memory[-1][j]
    return choose_memory
