"""`matflow.utils.py`"""

import collections
import copy
import itertools
import numpy as np
import random
import time


def parse_times(format_str):
    """Parse a string which contain time format code and one or
    more `%%r` to represent a random digit from 0 to 9."""

    time_parsed = time.strftime(format_str)
    rnd_all = ''
    while '%r' in time_parsed:
        rnd = str(random.randint(0, 9))
        rnd_all += rnd
        time_parsed = time_parsed.replace('%r', rnd, 1)

    return time_parsed, rnd_all


def zeropad(num, largest):
    """Return a zero-padded string of a number, given the largest number.

    TODO: want to support floating-point numbers as well? Or rename function
    accordingly.

    Parameters
    ----------
    num : int
        The number to be formatted with zeros padding on the left.
    largest : int
        The number that determines the number of zeros to pad with.

    Returns
    -------
    padded : str
        The original number, `num`, formatted as a string with zeros added
        on the left.

    """

    num_digits = len('{:.0f}'.format(largest))
    padded = '{0:0{width}}'.format(num, width=num_digits)

    return padded


def combine_list_of_dicts(a):

    a = copy.deepcopy(a)

    for i in range(1, len(a)):
        update_dict(a[0], a[i])

    return a[0]


def update_dict(base, upd):
    """Update an arbitrarily-nested dict."""

    for key, val in upd.items():
        if isinstance(base, collections.Mapping):
            if isinstance(val, collections.Mapping):
                r = update_dict(base.get(key, {}), val)
                base[key] = r
            else:
                base[key] = upd[key]
        else:
            base = {key: upd[key]}

    return base


def nest_lists(my_list):
    """
        `a` is a list of `N` sublists.

        E.g.
        my_list = [
            [1,2],
            [3,4,5],
            [6,7]
        ]

        returns a list of lists of length `N` such that all combinations of elements from sublists in
        `a` are found
        E.g
        out = [
            [1, 3, 6],
            [1, 3, 7],
            [1, 4, 6],
            [1, 4, 7],
            [1, 5, 6],
            [1, 5, 7],
            [2, 3, 6],
            [2, 3, 7],
            [2, 4, 6],
            [2, 4, 7],
            [2, 5, 6],
            [2, 5, 7]
        ]

    """

    N = len(my_list)
    sub_len = [len(i) for i in my_list]

    products = np.array([1] * (N + 1))
    for i in range(len(my_list) - 1, -1, -1):
        products[:i + 1] *= len(my_list[i])

    out = [[None for x in range(N)] for y in range(products[0])]

    for row_idx, row in enumerate(out):

        for col_idx, col in enumerate(row):

            num_repeats = products[col_idx + 1]
            sub_list_idx = int(row_idx / num_repeats) % len(my_list[col_idx])
            out[row_idx][col_idx] = copy.deepcopy(
                my_list[col_idx][sub_list_idx])

    return out


def repeat(lst, reps):
    'Repeat 1D list elements.'
    return list(itertools.chain.from_iterable(itertools.repeat(x, reps) for x in lst))


def tile(lst, tiles):
    'Tile a 1D list.'
    return lst * tiles


def index(lst, idx):
    'Get elements of a list.'
    return [lst[i] for i in idx]


def arange(size):
    'Get 1D list of increasing integers.'
    return list(range(size))
