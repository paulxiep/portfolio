from arithmetic_classes import *

'''
Special case: 3 numbers (3, 7, 5), fixed order.
            Parenthesis can be put over 3 and 5, or over 5 and 7,
            effectively either joins the first 2 terms first, or the last 2 terms first.
            Singular Operator can be put over Number or over Parenthesis (joined terms)
            Duo Operator must be put over 2 Entities.
'''


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
    if allow_factorial and number.is_positive() and number.value < 11 and int(number.value) == number.value:
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
        [n1, Operator.get('+'), n2],
        [n1, Operator.get('*'), n2]
    ] + [[n1, Operator.get('/'), n2]] * int(n2.value != 0))


def from_2(n1, n2):
    '''
    This function accepts 2 numbers.
    Returns all Duo operations on all Singular operations of 2 numbers.
    This is what you need in most cases.
    Note that this is a generator function.
    '''
    yield from (do for sn1 in singular_operators(n1)
                for sn2 in singular_operators(n2)
                for do in duo_operators(sn1, sn2))


def fixed_pos(*args):
    '''
    Accepts any number of numbers.
    It returns results of recursively matching all adjacent pairs of numbers with from_2,
    until only 1 number remains, in which case it wraps it in a Singular operator operation.
    Note that this is a generator function.
    '''
    if len(args) == 1:
        yield from singular_operators(args[0])
    elif len(args) > 1:
        yield from (fp
                    for i in range(len(args) - 1)
                    for pair in from_2(args[i], args[i + 1])
                    for fp in fixed_pos(*(list(args[:i]) + [pair] + list(args[i + 2:]))))
    else:
        raise NotImplementedError('0 argument in fixed_pos not implemented')


def search(*args, search_range=range(0, 101)):
    '''
    Search, for any number of number arguments, how to arrive at results within a search range.
    It is not advisable to go beyond 4 numbers unless you're willing to test your patience at waiting.
    '''
    out = {k: [] for k in search_range}

    for value in fixed_pos(*args):
        if value.value in search_range:
            out[int(value.value)].append(value.repr)

    for k, v in sorted(out.items()):
        if len(v) > 0:
            print(k, min(v, key=lambda x: x.count('(')))


if __name__ == '__main__':
    '''
    4 numbers is the maximum number of input advised.
    5 numbers will likely take too long for your patience.
    Define a search range as keyword argument if needed.
    '''
    search(3, 7, 5, 3)
