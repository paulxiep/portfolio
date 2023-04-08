def squash_idle(player_memory):
    '''
    used for squashing idle turns in memory
    '''
    new_memory = []
    for memory in player_memory:
        if not isinstance(memory[0], str) and not isinstance(memory[1], str):
            if not isinstance(memory[2], str):
                new_memory.append(list(memory))
            else:
                partial = [memory[0], memory[1]], list(memory[5:])
        else:
            if not isinstance(memory[2], str):
                new_memory.append(partial[0] + list(memory[2:5]) + partial[1])
            elif memory[4] == 1:
                new_memory.append(partial[0] + list(memory[2:5]) + partial[1])
    choose_memory = list(filter(lambda x: x[-1] in [0, 3], new_memory))
    # play_memory = list(filter(lambda x: x[-1] == 1, new_memory))
    for mem in [choose_memory]:
        if mem[-1][4] == 0:
            mem[-1][3] = new_memory[-1][3]
            mem[-1][4] = new_memory[-1][4]
    return choose_memory
