

def squash_idle(player_memory):
    '''
    used for squashing idle turns in memory
    '''
    new_memory = []
    for memory in player_memory:
        if not isinstance(memory[0], str) and not isinstance(memory[1], str):
            if not isinstance(memory[2], str):
                new_memory.append(list(memory[:5]))
            else:
                partial = [memory[0], memory[1]]
        else:
            if not isinstance(memory[2], str):
                new_memory.append(partial + list(memory[2:5]))
            elif memory[4] == 1:
                new_memory.append(partial + list(memory[2:5]))
    return new_memory