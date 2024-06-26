import numpy as np

def squash_idle(player_memory):
    '''
    used for squashing idle turns in memory
    '''
    choose_memory = []
    played_discarded = False
    for i, memory in enumerate(player_memory):
        if memory[-1] == 0 and not isinstance(memory[0], str):
            partial = [memory[0]], list(memory[8:]), memory[1]
        elif memory[-1] == 1 and not isinstance(memory[1], str):
            action = [partial[2] + 80 * memory[1]]
        elif memory[-1] == 2:
            if not isinstance(memory[2], str):
                choose_memory.append(partial[0] + action + [memory[2]] + list(memory[3:8]) + partial[1])
                played_discarded = True
            else:
                rewards = list(memory[3:8])
        elif memory[-1] == 3:
            if not played_discarded:
                choose_memory.append(partial[0] + action + [memory[2]] + rewards + partial[1])
            else:
                choose_memory.append(list(memory))
                played_discarded = False
    for j in range(3, 8):
        choose_memory[-1][j] = player_memory[-1][j]
    return choose_memory
