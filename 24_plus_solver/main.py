from arithmetic_classes import *
import time
from itertools import permutations
from functools import lru_cache

FACTORIAL_LIMIT = 8  # beyond this value, don't calculate factorial
MULTIPLY_LIMIT = 1800  # for results beyond this value (absolute), drop the multiplication results

def operation(*args):
    '''
    accepts 1, 2, or 3 arguments
    1 argument simply return the Number as is
    2 arguments is for PreSingularOperator + Number or Number + PostSingularOperator
    3 arguments is for Number + DuoOperator + Number
    '''
    if len(args) == 1:
        return args[0]
    elif len(args) == 2:
        return operation(args[0]) >> operation(args[1])
    elif len(args) == 3:
        return operation(args[0]) >> operation(args[1]) >> operation(args[2])
    else:
        print(args)
        raise NotImplementedError


def singular_operators(number, allow_factorial=True):
    '''
    Sqrt and factorial chaining not implemented.
    This function accepts a Number or base Python number
    Returns all defined Singular operations on it.
    Note that this is a generator function.
    '''
    if not isinstance(number, Number):
        number = Number.get(number)
    base = [[number],
            [Operator.get('-()'), number]]
    if number.integer_root():
        if number.is_positive():
            base += [[Operator.get('sqrt'), number], [Operator.get('-()'), operation(*[Operator.get('sqrt'), number])]]
        else:
            base += [[Operator.get('sqrt'), operation(*[Operator.get('-()'), number])], [Operator.get('-()'), operation(
                *[Operator.get('sqrt'), operation(*[Operator.get('-()'), number])])]]
    if allow_factorial and number.is_positive() and number.value < FACTORIAL_LIMIT+1 and float(number.value).is_integer():
        base += [[number, Operator.get('!')],
                 [Operator.get('-()'), operation(*[number, Operator.get('!')])]]
    yield from map(lambda x: operation(*x), base)


def duo_operators(n1, n2):
    '''
    NOT MEANT TO BE USED DIRECTLY, use from_2 unless you know what you're doing
    no need for -
    also check if divisor = 0
    This function accepts 2 numbers.
    Returns all Duo operations on 2 numbers
    Note that this is a generator function.
    '''
    yield from map(lambda x: operation(*x), [
        [n1, Operator.get('+'), n2]
    ] + [[n1, Operator.get('*'), n2]] * (abs(n1.value * n2.value) < MULTIPLY_LIMIT) \
      + [[n1, Operator.get('/'), n2]] * (int(n2.value != 0) and (n1.value/n2.value).is_integer()))


def from_2(n1, n2, allow_factorial=True):
    '''
    This function accepts 2 numbers.
    Returns all Duo operations on all Singular operations of 2 numbers.
    This is what you need in most cases.
    Note that this is a generator function.
    '''
    yield from (do for sn1 in singular_operators(n1, allow_factorial=allow_factorial)
                for sn2 in singular_operators(n2, allow_factorial=allow_factorial)
                for do in duo_operators(sn1, sn2))


def fixed_pos(*args, allow_factorial=True):
    '''
    Accepts any number of numbers.
    It returns results of recursively matching all adjacent pairs of numbers with from_2,
    until only 1 number remains, in which case it wraps it in a Singular operator operation.
    Note that this is a generator function.
    '''
    if len(args) == 1:
        yield from singular_operators(args[0], allow_factorial=allow_factorial)
    elif len(args) > 1:
        yield from (fp
                    for i in range(len(args) - 1)
                    for pair in from_2(args[i], args[i + 1], allow_factorial=allow_factorial)
                    for fp in fixed_pos(*(list(args[:i]) + [pair] + list(args[i + 2:])), allow_factorial=allow_factorial))
    else:
        raise NotImplementedError('0 argument in fixed_pos not implemented')

def permute_pos(*args, allow_factorial=False):
    '''
    Like fixed_pos, but with argument order permutation.
    Note that this is a generator function.
    '''
    yield from (fp for perm in permutations(args, len(args)) for fp in fixed_pos(*perm, allow_factorial=allow_factorial))

def search_range(*args, search_range=range(0, 51), allow_factorial=False):
    '''
    Search, for any number of number arguments, how to arrive at results within a search range.
    First try for fewer number of arguments, to get a feel for time required.
    Each argument added exponentially increases runtime.
    '''
    out = {k: None for k in search_range}
    i = 0
    for value in permute_pos(*args, allow_factorial=allow_factorial):
        if int(value.value) in search_range:
            if out[int(value.value)] is None:
                out[int(value.value)] = value.repr
                i += 1
                if i >= len(search_range):
                    break
            elif out[int(value.value)].count('(') > value.repr.count('('):
                out[int(value.value)] = value.repr

    for k, v in sorted(out.items()):
        print(k, ':', v)

def search_value(*args, search_value=729, allow_factorial=True):
    '''
    Like search_range, but search only for a single answer.
    First try for fewer number of arguments, to get a feel for time required.
    Each argument added exponentially increases runtime.
    '''
    for value in permute_pos(*args, allow_factorial=allow_factorial):
        if value.value == search_value:
            print(value.value, ':', value.repr)
            return value.value, value.repr


if __name__ == '__main__':
    '''
    First try for fewer number of arguments, to get a feel for time required. 
    Each argument added exponentially increases runtime.
    Setting allow_factorial=False also decreases runtime immensely.
    '''
    inputs = [3, 6, 4, 7, 5]
    cumulative = 0
    for i in range(101):
        start = time.time()
        search_value(*inputs, search_value=i, allow_factorial=True)
        temp = time.time() - start
        cumulative += temp
        print('took', temp, 'seconds')
    print('cumulative segregated searches took', cumulative, 'seconds')
    start = time.time()
    search_range(*inputs, search_range=range(0, 101), allow_factorial=True)
    print('combined search took', time.time() - start, 'seconds')
